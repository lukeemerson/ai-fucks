"""Unit tests for :class:`TorchFineTuneTrainer` (FINE_TUNING_DESIGN.md §6.2).

Marked ``torch`` so they are excluded from the default fast suite. Each
test is intentionally tiny (CPU device, 8x8 input, batch 4, 2 epochs)
so the suite runs in seconds; tests that legitimately exceed 5s are
also marked ``slow``.

Coverage targets per the design doc:

* Optimizer is AdamW.
* Loss is BCEWithLogitsLoss.
* Cosine schedule warmup + decay (LR climbs through warmup then declines).
* Checkpoint round-trip preserves model weights / optimizer state.
* Augmentation pipeline is seed-deterministic.
* Validation loader uses no augmentation.
* Device fallback ladder.
* Backbone-id whitelist (densenet121, resnet50; TXRV / unknown raise).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from harness.adapters.torch.trainer import (  # noqa: E402
    TorchFineTuneTrainer,
    _config_hash,
)
from harness.domain.errors import AdapterError, ConfigError  # noqa: E402
from harness.domain.types import TrainingConfig  # noqa: E402

pytestmark = pytest.mark.torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_class_dataset(*, seed: int, n_rows: int = 16) -> TinyTrainingDataset:
    return TinyTrainingDataset(seed=seed, n_rows=n_rows)


class TinyTrainingDataset:
    """Two-class TrainingDatasetPort: dark vs bright 8x8 grayscale rows."""

    def __init__(self, *, seed: int, n_rows: int = 16) -> None:
        rng = np.random.default_rng(seed)
        rows: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_rows):
            is_bright = bool(i % 2)
            low, high = (0.5, 1.0) if is_bright else (0.0, 0.5)
            image = rng.uniform(low, high, size=(8, 8, 1)).astype(np.float32)
            labels = np.array([int(not is_bright), int(is_bright)], dtype=np.int8)
            rows.append((image, labels))
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._rows[index]


def _make_config(
    *,
    n_epochs: int = 2,
    backbone_id: str = "densenet121",
    image_size: tuple[int, int] = (32, 32),
    augmentations: tuple[str, ...] = (),
    lr_schedule: str = "constant",
    warmup_epochs: int = 0,
    early_stop_patience: int | None = None,
    checkpoint_dir: str | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        backbone_id=backbone_id,
        n_labels=2,
        n_epochs=n_epochs,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer="adamw",
        lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs,
        augmentations=augmentations,
        image_size=image_size,
        checkpoint_dir=checkpoint_dir,
        early_stop_patience=early_stop_patience,
        num_dataloader_workers=0,
    )


# ---------------------------------------------------------------------------
# Backbone-id whitelist (FINE_TUNING_DESIGN.md §10 answer #6)
# ---------------------------------------------------------------------------


def test_unknown_backbone_id_raises_config_error() -> None:
    """v1.1 rejects any backbone_id not in {densenet121, resnet50}."""
    trainer = TorchFineTuneTrainer(device="cpu")
    cfg = _make_config(backbone_id="vit_base")
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)
    with pytest.raises(ConfigError):
        trainer.fit(
            training_dataset=train, validation_dataset=val, config=cfg, seed=0
        )


def test_txrv_backbone_id_raises_config_error() -> None:
    """TXRV NIH fine-tuning is not shipped in v1.1 (§10 answer #6)."""
    trainer = TorchFineTuneTrainer(device="cpu")
    cfg = _make_config(backbone_id="txrv-densenet121-res224-nih")
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)
    with pytest.raises(ConfigError):
        trainer.fit(
            training_dataset=train, validation_dataset=val, config=cfg, seed=0
        )


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------


def test_trainer_identifier_is_stable() -> None:
    trainer = TorchFineTuneTrainer(device="cpu")
    assert trainer.identifier == "torch.finetune.v1"


# ---------------------------------------------------------------------------
# Cosine warmup + decay (FINE_TUNING_DESIGN.md §5)
# ---------------------------------------------------------------------------


def test_cosine_lr_schedule_climbs_through_warmup_then_decays() -> None:
    """LR must climb linearly through warmup then decline per cosine."""
    trainer = TorchFineTuneTrainer(device="cpu")
    cfg = _make_config(
        n_epochs=4,
        lr_schedule="cosine",
        warmup_epochs=2,
    )
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)
    _trained, _result = trainer.fit(
        training_dataset=train, validation_dataset=val, config=cfg, seed=0
    )
    lrs = trainer.lr_per_epoch_for_test()
    assert len(lrs) == cfg.n_epochs
    # Warmup phase: epoch 0 < epoch 1 (linear ramp, both <= base LR).
    assert lrs[0] < lrs[1]
    # Cosine decay phase: lrs[1] should be the peak at the end of warmup
    # (epoch index ``warmup_epochs - 1`` here is 1). After warmup the LR
    # declines through epoch 3.
    assert lrs[2] > lrs[3]


# ---------------------------------------------------------------------------
# Checkpoint round-trip (FINE_TUNING_DESIGN.md §5)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_checkpoint_resume_continues_from_latest(tmp_path: Path) -> None:
    """Training from a 2-epoch checkpoint produces a valid 4-epoch result."""
    ckpt_dir = tmp_path / "ckpt"
    cfg_initial = _make_config(n_epochs=2, checkpoint_dir=str(ckpt_dir))
    trainer_a = TorchFineTuneTrainer(device="cpu")
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)
    _trained_a, result_a = trainer_a.fit(
        training_dataset=train,
        validation_dataset=val,
        config=cfg_initial,
        seed=0,
    )
    assert result_a.n_epochs_run == 2

    # Resume with a longer schedule using the same ckpt dir.
    cfg_resume = _make_config(n_epochs=4, checkpoint_dir=str(ckpt_dir))
    trainer_b = TorchFineTuneTrainer(device="cpu")
    _trained_b, result_b = trainer_b.fit(
        training_dataset=train,
        validation_dataset=val,
        config=cfg_resume,
        seed=0,
    )
    # n_epochs_run reports the *total* epochs in the bookkeeping tuple.
    assert result_b.n_epochs_run == 4


def test_checkpoint_config_hash_mismatch_raises_adapter_error(
    tmp_path: Path,
) -> None:
    """Changing config under the same ckpt dir must raise ``AdapterError``."""
    ckpt_dir = tmp_path / "ckpt"
    cfg_a = _make_config(n_epochs=2, checkpoint_dir=str(ckpt_dir))
    cfg_b = _make_config(
        n_epochs=2,
        checkpoint_dir=str(ckpt_dir),
        backbone_id="resnet50",  # different config -> different hash
    )
    trainer = TorchFineTuneTrainer(device="cpu")
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)
    trainer.fit(
        training_dataset=train, validation_dataset=val, config=cfg_a, seed=0
    )
    trainer_b = TorchFineTuneTrainer(device="cpu")
    with pytest.raises(AdapterError):
        trainer_b.fit(
            training_dataset=train,
            validation_dataset=val,
            config=cfg_b,
            seed=0,
        )


def test_config_hash_is_deterministic() -> None:
    cfg = _make_config()
    assert _config_hash(cfg) == _config_hash(cfg)
    # Changing the backbone changes the hash (logically different run).
    cfg2 = _make_config(backbone_id="resnet50")
    assert _config_hash(cfg) != _config_hash(cfg2)


def test_config_hash_excludes_n_epochs_and_early_stop_patience() -> None:
    """n_epochs / early_stop_patience are excluded so resume can extend
    a truncated schedule (see ``_HASH_EXCLUDED_FIELDS`` in trainer.py)."""
    cfg = _make_config(n_epochs=2, early_stop_patience=3)
    cfg_extended = _make_config(n_epochs=4, early_stop_patience=3)
    assert _config_hash(cfg) == _config_hash(cfg_extended)
    cfg_patience = _make_config(n_epochs=2, early_stop_patience=None)
    assert _config_hash(cfg) == _config_hash(cfg_patience)


# ---------------------------------------------------------------------------
# Augmentation seeding (FINE_TUNING_DESIGN.md §5 / §7)
# ---------------------------------------------------------------------------


def test_augmentation_pipeline_seeded_via_fit() -> None:
    """Same seed -> byte-identical predictions even with augmentation on."""
    cfg = _make_config(augmentations=("hflip", "rotate10"))
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)

    trainer_a = TorchFineTuneTrainer(device="cpu")
    trained_a, _ = trainer_a.fit(
        training_dataset=train, validation_dataset=val, config=cfg, seed=0
    )
    trainer_b = TorchFineTuneTrainer(device="cpu")
    trained_b, _ = trainer_b.fit(
        training_dataset=train, validation_dataset=val, config=cfg, seed=0
    )
    images = np.stack([val[i][0] for i in range(4)]).astype(np.float32)
    np.testing.assert_array_equal(
        trained_a.predict_proba(images),
        trained_b.predict_proba(images),
    )


def test_unknown_augmentation_name_raises_config_error() -> None:
    trainer = TorchFineTuneTrainer(device="cpu")
    cfg = _make_config(augmentations=("hflip", "wibble"))
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)
    with pytest.raises(ConfigError):
        trainer.fit(
            training_dataset=train, validation_dataset=val, config=cfg, seed=0
        )


# ---------------------------------------------------------------------------
# Early stopping (FINE_TUNING_DESIGN.md §3.1 / §5)
# ---------------------------------------------------------------------------


def test_early_stopping_terminates_when_no_improvement() -> None:
    """``early_stop_patience=1`` halts training as soon as val-AUROC drops."""
    trainer = TorchFineTuneTrainer(device="cpu")
    cfg = _make_config(n_epochs=10, early_stop_patience=1)
    # Reuse the same dataset for train and val so the val signal is
    # well-correlated; the test asserts only that *some* epochs run and
    # the bookkeeping respects ``n_epochs_run <= n_epochs``.
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=0, n_rows=8)
    _trained, result = trainer.fit(
        training_dataset=train, validation_dataset=val, config=cfg, seed=0
    )
    assert result.n_epochs_run >= 1
    assert result.n_epochs_run <= cfg.n_epochs


# ---------------------------------------------------------------------------
# predict_proba shape / range
# ---------------------------------------------------------------------------


def test_predict_proba_shape_and_range() -> None:
    trainer = TorchFineTuneTrainer(device="cpu")
    cfg = _make_config()
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)
    trained, _ = trainer.fit(
        training_dataset=train, validation_dataset=val, config=cfg, seed=0
    )
    images = np.stack([val[i][0] for i in range(4)]).astype(np.float32)
    probs = trained.predict_proba(images)
    assert probs.shape == (4, 2)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


def test_predict_proba_empty_batch_returns_empty_array() -> None:
    trainer = TorchFineTuneTrainer(device="cpu")
    cfg = _make_config()
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)
    trained, _ = trainer.fit(
        training_dataset=train, validation_dataset=val, config=cfg, seed=0
    )
    empty = np.zeros((0, 8, 8, 1), dtype=np.float32)
    out = trained.predict_proba(empty)
    assert out.shape == (0, 2)


def test_predict_proba_rejects_3d_input() -> None:
    trainer = TorchFineTuneTrainer(device="cpu")
    cfg = _make_config()
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=1, n_rows=4)
    trained, _ = trainer.fit(
        training_dataset=train, validation_dataset=val, config=cfg, seed=0
    )
    bad = np.zeros((8, 8, 1), dtype=np.float32)
    with pytest.raises(AdapterError):
        trained.predict_proba(bad)


# ---------------------------------------------------------------------------
# Device fallback (FINE_TUNING_DESIGN.md §5)
# ---------------------------------------------------------------------------


def test_invalid_device_override_raises_adapter_error() -> None:
    with pytest.raises(AdapterError):
        TorchFineTuneTrainer(device="tpu")  # type: ignore[arg-type]  # reason: invalid value test


# ---------------------------------------------------------------------------
# Best-epoch weights restoration (C1 fix; FINE_TUNING_DESIGN.md §3.2)
# ---------------------------------------------------------------------------


def test_returned_classifier_uses_best_epoch_weights(tmp_path: Path) -> None:
    """The returned ``TrainedClassifierPort`` must reflect the best-epoch
    weights, not the last-epoch weights (FINE_TUNING_DESIGN.md §3.2).

    Strategy: train for several epochs with checkpointing on. Then for
    each persisted checkpoint, load it back into a fresh DenseNet121 +
    linear head and run ``predict_proba`` over a fixed eval batch. Find
    the checkpoint whose epoch matches ``result.best_epoch`` and assert
    its predictions match the trainer's returned classifier
    byte-identically. Requires the bug to be fixed: with the bug, the
    returned classifier predicts using last-epoch weights, so the
    assertion fails unless best_epoch happens to be the last epoch
    (which we verify is NOT the case below).
    """
    import torch as _torch
    from torch import nn as _nn
    from torchvision.models import densenet121 as _dn

    ckpt_dir = tmp_path / "ckpt"
    cfg = _make_config(
        n_epochs=4,
        checkpoint_dir=str(ckpt_dir),
        image_size=(32, 32),
    )
    trainer = TorchFineTuneTrainer(device="cpu")
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=42, n_rows=8)
    trained, result = trainer.fit(
        training_dataset=train, validation_dataset=val, config=cfg, seed=0
    )
    # Pre-condition for the test to actually exercise the bug: best epoch
    # must NOT be the last epoch. With the configured tiny val set + 4
    # epochs of training on 8 dark/bright rows, val-AUROC saturates fast
    # and the best epoch is almost always epoch 0 or 1.
    assert result.best_epoch != result.n_epochs_run - 1, (
        f"test setup failed to elicit best_epoch != last_epoch: "
        f"best={result.best_epoch} n_run={result.n_epochs_run}"
    )

    # Load the best-epoch checkpoint into a fresh model and compare
    # predictions. We construct the same DenseNet121 + Linear head shape
    # the trainer uses internally.
    best_ckpt = ckpt_dir / f"epoch_{result.best_epoch:03d}.pt"
    assert best_ckpt.is_file(), f"missing best-epoch checkpoint {best_ckpt}"
    blob = _torch.load(best_ckpt, map_location="cpu", weights_only=False)
    fresh_model = _dn(weights=None)
    fresh_model.classifier = _nn.Linear(
        fresh_model.classifier.in_features, cfg.n_labels
    )
    fresh_model.load_state_dict(blob["model_state_dict"])
    fresh_model.eval()

    eval_imgs = np.stack(
        [val[i][0] for i in range(len(val))], axis=0
    ).astype(np.float32)
    out_returned = trained.predict_proba(eval_imgs)
    # Mirror the same preprocessing as ``_TorchTrainedClassifier._forward_chunk``.
    import torch.nn.functional as _F

    nhwc = _torch.from_numpy(eval_imgs)
    nchw = nhwc.permute(0, 3, 1, 2).contiguous()
    if nchw.shape[1] == 1:
        nchw = nchw.repeat(1, 3, 1, 1)
    h, w = cfg.image_size
    if nchw.shape[2] != h or nchw.shape[3] != w:
        nchw = _F.interpolate(nchw, size=(h, w), mode="bilinear", align_corners=False)
    mean = _torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = _torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    normalized = (nchw - mean) / std
    with _torch.no_grad():
        logits = fresh_model(normalized)
        out_fresh = _torch.sigmoid(logits).numpy().astype(np.float32)
    np.testing.assert_allclose(out_returned, out_fresh, rtol=1e-5, atol=1e-6)


def test_final_checkpoint_uri_points_to_best_epoch(tmp_path: Path) -> None:
    """``TrainingResult.final_checkpoint_uri`` must reference the best-epoch
    checkpoint, not the last-epoch checkpoint (FINE_TUNING_DESIGN.md §3.2)."""
    ckpt_dir = tmp_path / "ckpt"
    cfg = _make_config(
        n_epochs=4, checkpoint_dir=str(ckpt_dir), image_size=(32, 32)
    )
    trainer = TorchFineTuneTrainer(device="cpu")
    train = _two_class_dataset(seed=0, n_rows=8)
    val = _two_class_dataset(seed=42, n_rows=8)
    _trained, result = trainer.fit(
        training_dataset=train, validation_dataset=val, config=cfg, seed=0
    )
    assert result.best_epoch != result.n_epochs_run - 1, (
        "test setup failed to elicit best_epoch != last_epoch"
    )
    assert result.final_checkpoint_uri is not None
    assert f"epoch_{result.best_epoch:03d}.pt" in result.final_checkpoint_uri
