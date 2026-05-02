"""Validator tests for ``TrainingConfig`` and ``TrainingResult``.

Written RED-first under TDD discipline before the ``__post_init__``
implementations land in ``harness/domain/types.py``. The validation rules
are taken verbatim from FINE_TUNING_DESIGN.md §3.1, with the v1.1
optimizer literal narrowed to ``{"adamw"}`` per the §3/§5 reconciliation
documented in this PR (see FINE_TUNING_DESIGN.md §3.1).

The tests intentionally only check construction-time validation; the
trainer-adapter behaviour is exercised by the contract suite and
adapter unit tests in separate files.
"""

from __future__ import annotations

import pytest

from harness.domain.errors import ConfigError
from harness.domain.types import (
    ExperimentConfig,
    TrainingConfig,
    TrainingResult,
)


def _make_valid_config(**overrides: object) -> TrainingConfig:
    """Build a minimally valid TrainingConfig; tests override individual fields."""
    base: dict[str, object] = {
        "backbone_id": "torchvision.densenet121",
        "n_labels": 14,
        "n_epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "warmup_epochs": 1,
        "augmentations": ("hflip", "rotate10"),
        "image_size": (224, 224),
        "checkpoint_dir": None,
        "early_stop_patience": 3,
        "num_dataloader_workers": 0,
    }
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]  # reason: helper builds via kwargs


# ---------------------------------------------------------------------------
# TrainingConfig validators (FINE_TUNING_DESIGN.md §3.1)
# ---------------------------------------------------------------------------


def test_training_config_accepts_valid_fields() -> None:
    cfg = _make_valid_config()
    assert cfg.backbone_id == "torchvision.densenet121"
    assert cfg.n_epochs == 2


def test_training_config_rejects_zero_n_labels() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(n_labels=0)


def test_training_config_rejects_negative_n_labels() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(n_labels=-1)


def test_training_config_rejects_zero_n_epochs() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(n_epochs=0)


def test_training_config_rejects_zero_batch_size() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(batch_size=0)


def test_training_config_rejects_zero_learning_rate() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(learning_rate=0.0)


def test_training_config_rejects_negative_learning_rate() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(learning_rate=-1e-3)


def test_training_config_rejects_negative_weight_decay() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(weight_decay=-0.01)


def test_training_config_accepts_zero_weight_decay() -> None:
    cfg = _make_valid_config(weight_decay=0.0)
    assert cfg.weight_decay == 0.0


def test_training_config_rejects_unknown_optimizer() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(optimizer="nadam")


def test_training_config_rejects_sgd_optimizer_in_v1_1() -> None:
    """v1.1 narrows optimizer to ``{"adamw"}`` (FINE_TUNING_DESIGN.md §3.1)."""
    with pytest.raises(ConfigError):
        _make_valid_config(optimizer="sgd")


def test_training_config_rejects_unknown_lr_schedule() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(lr_schedule="exponential")


def test_training_config_accepts_constant_lr_schedule() -> None:
    cfg = _make_valid_config(lr_schedule="constant", warmup_epochs=0)
    assert cfg.lr_schedule == "constant"


def test_training_config_rejects_negative_warmup_epochs() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(warmup_epochs=-1)


def test_training_config_rejects_warmup_greater_than_n_epochs() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(n_epochs=2, warmup_epochs=3)


def test_training_config_accepts_zero_warmup_epochs() -> None:
    cfg = _make_valid_config(warmup_epochs=0)
    assert cfg.warmup_epochs == 0


def test_training_config_rejects_zero_image_height() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(image_size=(0, 224))


def test_training_config_rejects_zero_image_width() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(image_size=(224, 0))


def test_training_config_rejects_zero_early_stop_patience() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(early_stop_patience=0)


def test_training_config_rejects_negative_early_stop_patience() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(early_stop_patience=-1)


def test_training_config_accepts_none_early_stop_patience() -> None:
    cfg = _make_valid_config(early_stop_patience=None)
    assert cfg.early_stop_patience is None


def test_training_config_rejects_nonzero_dataloader_workers() -> None:
    """v1.1 locks ``num_dataloader_workers`` to 0 (ARCHITECTURE.md §9)."""
    with pytest.raises(ConfigError):
        _make_valid_config(num_dataloader_workers=1)


def test_training_config_rejects_negative_dataloader_workers() -> None:
    with pytest.raises(ConfigError):
        _make_valid_config(num_dataloader_workers=-1)


# ---------------------------------------------------------------------------
# TrainingResult validators (FINE_TUNING_DESIGN.md §3.2)
# ---------------------------------------------------------------------------


def _make_valid_result(**overrides: object) -> TrainingResult:
    base: dict[str, object] = {
        "n_epochs_run": 2,
        "train_loss_per_epoch": (0.5, 0.4),
        "val_loss_per_epoch": (0.5, 0.45),
        "val_macro_auroc_per_epoch": (0.6, 0.65),
        "best_epoch": 1,
        "final_checkpoint_uri": None,
    }
    base.update(overrides)
    return TrainingResult(**base)  # type: ignore[arg-type]  # reason: helper builds via kwargs


def test_training_result_accepts_valid_fields() -> None:
    result = _make_valid_result()
    assert result.n_epochs_run == 2
    assert result.best_epoch == 1


def test_training_result_rejects_zero_n_epochs_run() -> None:
    with pytest.raises(ConfigError):
        _make_valid_result(
            n_epochs_run=0,
            train_loss_per_epoch=(),
            val_loss_per_epoch=(),
            val_macro_auroc_per_epoch=(),
            best_epoch=0,
        )


def test_training_result_rejects_negative_n_epochs_run() -> None:
    with pytest.raises(ConfigError):
        _make_valid_result(n_epochs_run=-1)


def test_training_result_rejects_loss_length_mismatch() -> None:
    with pytest.raises(ConfigError):
        _make_valid_result(
            n_epochs_run=2,
            train_loss_per_epoch=(0.5,),  # length 1, n_epochs_run=2
        )


def test_training_result_rejects_val_loss_length_mismatch() -> None:
    with pytest.raises(ConfigError):
        _make_valid_result(
            n_epochs_run=2,
            val_loss_per_epoch=(0.5,),
        )


def test_training_result_rejects_auroc_length_mismatch() -> None:
    with pytest.raises(ConfigError):
        _make_valid_result(
            n_epochs_run=2,
            val_macro_auroc_per_epoch=(0.6,),
        )


def test_training_result_rejects_best_epoch_out_of_range() -> None:
    with pytest.raises(ConfigError):
        _make_valid_result(n_epochs_run=2, best_epoch=2)


def test_training_result_rejects_negative_best_epoch() -> None:
    with pytest.raises(ConfigError):
        _make_valid_result(best_epoch=-1)


# ---------------------------------------------------------------------------
# ExperimentConfig.training field (FINE_TUNING_DESIGN.md §4.4)
# ---------------------------------------------------------------------------


def _make_experiment_config(**overrides: object) -> ExperimentConfig:
    from harness.domain.types import BootstrapConfig, ThresholdConfig

    base: dict[str, object] = {
        "experiment_name": "test",
        "dataset_name": "fake",
        "label_names": ("a", "b"),
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "seed": 0,
        "bootstrap": BootstrapConfig(n_resamples=8, confidence=0.95, seed=1),
        "threshold": ThresholdConfig(
            method="pr_sweep", shrinkage=0.5, clamp_lo=0.05, clamp_hi=0.95
        ),
        "backbone_id": "fake",
        "head_id": "fake",
        "calibrator_id": "fake",
        "artifact_root": "memory://t",
        "notes": "",
    }
    base.update(overrides)
    return ExperimentConfig(**base)  # type: ignore[arg-type]


def test_experiment_config_training_defaults_to_none() -> None:
    cfg = _make_experiment_config()
    assert cfg.training is None


def test_experiment_config_accepts_training_subconfig() -> None:
    training = _make_valid_config(n_labels=2)
    cfg = _make_experiment_config(training=training)
    assert cfg.training is not None
    assert cfg.training.n_epochs == 2
