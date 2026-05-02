"""Composition root for harness experiments.

Two named entry points (FINE_TUNING_DESIGN.md §4):

* :func:`run_experiment` -- frozen-feature pipeline (backbone + head are
  pre-trained / pre-fit). Used by :func:`harness.composition.factories.build_publication_runner_v1`.
* :func:`run_finetune_experiment` -- end-to-end fine-tune pipeline (the
  trainer subsumes both backbone and head). Used by
  :func:`harness.composition.factories.build_finetune_runner_v1`.

The two runners share calibrator + threshold + metrics + artifact-store
wiring (steps 8-12 of FINE_TUNING_DESIGN.md §4.2); the fork is only between
"extract features" and "train end-to-end."

Per ARCHITECTURE.md section 6 the runner is fully deterministic given
``config`` and the supplied ports. It derives every sub-seed via
``randomness.child_seed`` with stable labels.

Determinism note
----------------
The ``clock`` keyword on :func:`run_experiment` is optional. When it is
``None`` the runner stamps the produced ``ModelCard`` with a fixed Unix-epoch
``datetime(1970, 1, 1, tzinfo=UTC)`` placeholder rather than wall-clock
``datetime.now``. This keeps the end-to-end pipeline reproducible (the model
card hash is stable across runs) without requiring every caller to wire a
clock adapter. Production callers that need a real timestamp must pass an
explicit ``datetime`` for ``clock``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from harness.composition._finetune_pipeline import (
    ImageDecoder,
    _InMemoryTrainingDataset,
)
from harness.domain.errors import ConfigError, ContractViolation
from harness.domain.types import (
    Dataset,
    ExperimentConfig,
    ExperimentResult,
    MetricReport,
    ModelCard,
    Probabilities,
    Sample,
    Split,
    ThresholdSet,
    TrainingResult,
)
from harness.ports.artifact_store import ArtifactStorePort
from harness.ports.backbone import BackbonePort
from harness.ports.calibrator import CalibratorPort
from harness.ports.classifier_head import ClassifierHeadPort
from harness.ports.dataset import DatasetPort
from harness.ports.metrics import MetricsPort
from harness.ports.randomness import RandomnessPort
from harness.ports.splitter import SplitterPort
from harness.ports.threshold import ThresholdPort
from harness.ports.trainer import TrainedClassifierPort, TrainerPort

__all__ = [
    "BYTES_IMAGE_SHAPE",
    "run_experiment",
    "run_finetune_experiment",
]

# Default image shape used when materialising raw bytes into an NHWC tensor for
# the backbone. Composition factories that wire fakes choose a backbone whose
# ``embedding_dim`` matches this shape.
BYTES_IMAGE_SHAPE: tuple[int, int, int] = (4, 4, 2)

# Deterministic placeholder used for ``ModelCard.created_at`` when no ``clock``
# is supplied. Chosen so that two runs with identical config and identical
# ports produce byte-identical model cards.
_EPOCH: datetime = datetime(1970, 1, 1, tzinfo=UTC)


def run_experiment(
    config: ExperimentConfig,
    *,
    dataset: DatasetPort,
    splitter: SplitterPort,
    backbone: BackbonePort,
    head: ClassifierHeadPort,
    calibrator: CalibratorPort,
    thresholds: ThresholdPort,
    metrics: MetricsPort,
    store: ArtifactStorePort,
    randomness: RandomnessPort,
    clock: datetime | None = None,
) -> ExperimentResult:
    """Run one full experiment, end-to-end, against the provided ports.

    When ``clock`` is ``None`` the resulting :class:`ModelCard.created_at`
    field is set to the Unix epoch (1970-01-01T00:00:00Z) -- a deterministic
    placeholder that keeps the run reproducible. Production callers that need
    a real timestamp must pass one explicitly.

    Per FINE_TUNING_DESIGN.md §4.4 ``config.training`` must be ``None`` here;
    fine-tune runs flow through :func:`run_finetune_experiment` instead.
    """
    if config.training is not None:
        raise ConfigError(
            "run_experiment requires config.training is None; fine-tune "
            "runs flow through run_finetune_experiment "
            "(FINE_TUNING_DESIGN.md §4.4)"
        )
    randomness.seed_all(config.seed)

    ds = dataset.load()
    if ds.label_names != config.label_names:
        raise ContractViolation(
            f"dataset.label_names {ds.label_names!r} != "
            f"config.label_names {config.label_names!r}"
        )

    split_seed = randomness.child_seed(config.seed, "split")
    split = splitter.split(
        ds,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=split_seed,
    )

    train_features = _extract_features(ds, split.train_indices, dataset, backbone)
    val_features = _extract_features(ds, split.val_indices, dataset, backbone)
    test_features = _extract_features(ds, split.test_indices, dataset, backbone)

    train_labels = _labels_array(ds, split.train_indices)
    val_labels = _labels_array(ds, split.val_indices)

    head.fit(train_features, train_labels)
    val_raw = head.predict_proba(val_features)
    test_raw = head.predict_proba(test_features)

    calibrator.fit(val_raw, val_labels)
    val_calibrated_arr = calibrator.transform(val_raw)
    test_calibrated_arr = calibrator.transform(test_raw)

    val_probs = _build_probabilities(ds, split.val_indices, val_calibrated_arr)
    test_probs = _build_probabilities(ds, split.test_indices, test_calibrated_arr)

    val_label_seq = _labels_as_sequence(ds, split.val_indices)
    test_label_seq = _labels_as_sequence(ds, split.test_indices)

    threshold_set = thresholds.fit(
        val_probs, val_label_seq, config=config.threshold
    )
    test_predictions = thresholds.apply(test_probs, threshold_set)
    report = metrics.evaluate(
        test_probs, test_label_seq, threshold_set, bootstrap=config.bootstrap
    )

    created_at = clock if clock is not None else _EPOCH
    model_card = _build_model_card(
        config=config,
        split=split,
        threshold_set=threshold_set,
        report=report,
        backbone_id=_describe_port(backbone, "backbone_id", config.backbone_id),
        head_id=_describe_port(head, "head_id", config.head_id),
        calibrator_id=_describe_port(calibrator, "calibrator_id", config.calibrator_id),
        created_at=created_at,
        notes=config.notes,
    )

    artifact_uris = {
        "model_card": store.write_model_card(model_card),
        "thresholds": store.write_thresholds(threshold_set),
        "test_predictions": store.write_predictions(test_predictions, "test"),
        "metric_report": store.write_metric_report(report),
    }

    return ExperimentResult(
        config=config,
        split=split,
        thresholds=threshold_set,
        val_probabilities=val_probs,
        test_probabilities=test_probs,
        test_predictions=test_predictions,
        report=report,
        model_card=model_card,
        artifact_uris=artifact_uris,
    )


# ---------------------------------------------------------------------------
# v1.1 fine-tune runner (FINE_TUNING_DESIGN.md §4.2)
# ---------------------------------------------------------------------------


def run_finetune_experiment(
    config: ExperimentConfig,
    *,
    dataset: DatasetPort,
    splitter: SplitterPort,
    trainer: TrainerPort,
    calibrator: CalibratorPort,
    thresholds: ThresholdPort,
    metrics: MetricsPort,
    store: ArtifactStorePort,
    randomness: RandomnessPort,
    decoder: ImageDecoder,
    clock: datetime | None = None,
) -> ExperimentResult:
    """Run one full fine-tune experiment, end-to-end (FINE_TUNING_DESIGN.md §4.2).

    Mirrors :func:`run_experiment` for steps 8-12 (calibrator -> threshold ->
    metrics -> artifacts). The fork is between feature-extraction (frozen path)
    and end-to-end training (this path):

    * Builds an :class:`_InMemoryTrainingDataset` per split via the supplied
      ``decoder`` callable (which decodes the dataset's raw bytes into a
      ``(H, W, C) float32`` array).
    * Calls :meth:`TrainerPort.fit` with the train + val datasets and a
      derived sub-seed.
    * Runs :meth:`TrainedClassifierPort.predict_proba` over val/test images
      and threads the probabilities through the existing
      calibrator/threshold/metrics chain unchanged.

    ``config.training`` must be set; ``None`` raises :class:`ConfigError`.
    """
    if config.training is None:
        raise ConfigError(
            "run_finetune_experiment requires config.training; for "
            "frozen-feature runs use run_experiment "
            "(FINE_TUNING_DESIGN.md §4.4)"
        )
    training_cfg = config.training
    if training_cfg.n_labels != len(config.label_names):
        raise ConfigError(
            f"config.training.n_labels {training_cfg.n_labels} != "
            f"len(config.label_names) {len(config.label_names)}"
        )
    randomness.seed_all(config.seed)
    ds = dataset.load()
    if ds.label_names != config.label_names:
        raise ContractViolation(
            f"dataset.label_names {ds.label_names!r} != "
            f"config.label_names {config.label_names!r}"
        )
    split_seed = randomness.child_seed(config.seed, "split")
    trainer_seed = randomness.child_seed(config.seed, "trainer")
    split = splitter.split(
        ds,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=split_seed,
    )
    image_size = training_cfg.image_size
    n_labels = training_cfg.n_labels
    train_td = _InMemoryTrainingDataset(
        source=dataset,
        indices=split.train_indices,
        n_labels=n_labels,
        image_size=image_size,
        decoder=decoder,
    )
    val_td = _InMemoryTrainingDataset(
        source=dataset,
        indices=split.val_indices,
        n_labels=n_labels,
        image_size=image_size,
        decoder=decoder,
    )
    test_td = _InMemoryTrainingDataset(
        source=dataset,
        indices=split.test_indices,
        n_labels=n_labels,
        image_size=image_size,
        decoder=decoder,
    )
    trained, training_result = trainer.fit(
        training_dataset=train_td,
        validation_dataset=val_td,
        config=training_cfg,
        seed=trainer_seed,
    )
    val_raw = _stack_predict(trained, val_td)
    test_raw = _stack_predict(trained, test_td)
    val_labels = _labels_array(ds, split.val_indices)
    calibrator.fit(val_raw, val_labels)
    val_calibrated_arr = calibrator.transform(val_raw)
    test_calibrated_arr = calibrator.transform(test_raw)
    val_probs = _build_probabilities(ds, split.val_indices, val_calibrated_arr)
    test_probs = _build_probabilities(ds, split.test_indices, test_calibrated_arr)
    val_label_seq = _labels_as_sequence(ds, split.val_indices)
    test_label_seq = _labels_as_sequence(ds, split.test_indices)
    threshold_set = thresholds.fit(
        val_probs, val_label_seq, config=config.threshold
    )
    test_predictions = thresholds.apply(test_probs, threshold_set)
    report = metrics.evaluate(
        test_probs, test_label_seq, threshold_set, bootstrap=config.bootstrap
    )
    created_at = clock if clock is not None else _EPOCH
    notes = _build_finetune_notes(
        notes=config.notes, trainer=trainer, training_result=training_result
    )
    model_card = _build_model_card(
        config=config,
        split=split,
        threshold_set=threshold_set,
        report=report,
        backbone_id=_describe_port(trainer, "backbone_id", config.backbone_id),
        head_id=_describe_port(trainer, "head_id", config.head_id),
        calibrator_id=_describe_port(
            calibrator, "calibrator_id", config.calibrator_id
        ),
        created_at=created_at,
        notes=notes,
    )
    artifact_uris = {
        "model_card": store.write_model_card(model_card),
        "thresholds": store.write_thresholds(threshold_set),
        "test_predictions": store.write_predictions(test_predictions, "test"),
        "metric_report": store.write_metric_report(report),
    }
    return ExperimentResult(
        config=config,
        split=split,
        thresholds=threshold_set,
        val_probabilities=val_probs,
        test_probabilities=test_probs,
        test_predictions=test_predictions,
        report=report,
        model_card=model_card,
        artifact_uris=artifact_uris,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _selected_samples(ds: Dataset, indices: Sequence[int]) -> tuple[Sample, ...]:
    return tuple(ds.samples[i] for i in indices)


def _extract_features(
    ds: Dataset,
    indices: Sequence[int],
    dataset: DatasetPort,
    backbone: BackbonePort,
) -> NDArray[np.float32]:
    samples = _selected_samples(ds, indices)
    if not samples:
        return np.zeros((0, backbone.embedding_dim), dtype=np.float32)
    image_batch = _bytes_to_image_tensor(
        tuple(dataset.get_image_bytes(s.image_ref) for s in samples)
    )
    features = backbone.extract(image_batch)
    return features.astype(np.float32, copy=False)


def _bytes_to_image_tensor(byte_blobs: Sequence[bytes]) -> NDArray[np.float32]:
    h, w, c = BYTES_IMAGE_SHAPE
    n_bytes = h * w * c
    n = len(byte_blobs)
    out = np.zeros((n, h, w, c), dtype=np.float32)
    for i, blob in enumerate(byte_blobs):
        if len(blob) >= n_bytes:
            material = blob[:n_bytes]
        else:
            # Pad deterministically by hashing again -- keeps the tensor
            # the same shape for all callers without relying on caller-side
            # knowledge of the image shape.
            extra = hashlib.sha256(blob).digest()
            material = (blob + extra)[:n_bytes]
        arr = np.frombuffer(material, dtype=np.uint8).astype(np.float32)
        out[i] = (arr / 255.0).reshape(h, w, c)
    return out


def _labels_array(ds: Dataset, indices: Sequence[int]) -> NDArray[np.int8]:
    n_labels = len(ds.label_names)
    if not indices:
        return np.zeros((0, n_labels), dtype=np.int8)
    rows = [ds.samples[i].labels for i in indices]
    return np.asarray(rows, dtype=np.int8)


def _labels_as_sequence(
    ds: Dataset, indices: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in ds.samples[i].labels) for i in indices)


def _build_probabilities(
    ds: Dataset,
    indices: Sequence[int],
    values: NDArray[np.float32],
) -> Probabilities:
    sample_ids = tuple(ds.samples[i].sample_id for i in indices)
    clipped = np.clip(values.astype(np.float32, copy=False), 0.0, 1.0)
    return Probabilities(
        sample_ids=sample_ids,
        label_names=ds.label_names,
        values=clipped,
    )


def _stack_predict(
    trained: TrainedClassifierPort, td: _InMemoryTrainingDataset
) -> NDArray[np.float32]:
    """Stack ``td``'s decoded images into NHWC and run ``predict_proba``."""
    if len(td) == 0:
        return np.zeros((0, trained.n_labels), dtype=np.float32)
    images = np.stack([td[i][0] for i in range(len(td))]).astype(
        np.float32, copy=False
    )
    return trained.predict_proba(images)


def _describe_port(port: object, attr: str, fallback: str) -> str:
    """Best-effort textual identifier for a port -- prefers ``identifier`` prop."""
    identifier = getattr(port, "identifier", None)
    if isinstance(identifier, str) and identifier:
        return identifier
    cfg_value = getattr(port, attr, None)
    if isinstance(cfg_value, str) and cfg_value:
        return cfg_value
    return fallback


def _build_model_card(
    *,
    config: ExperimentConfig,
    split: Split,
    threshold_set: ThresholdSet,
    report: MetricReport,
    backbone_id: str,
    head_id: str,
    calibrator_id: str,
    created_at: datetime,
    notes: str,
) -> ModelCard:
    return ModelCard(
        name=config.experiment_name,
        version="v1",
        created_at=created_at,
        backbone=backbone_id,
        head=head_id,
        calibrator=calibrator_id,
        threshold_method=threshold_set.method,
        label_names=config.label_names,
        train_size=len(split.train_indices),
        val_size=len(split.val_indices),
        test_size=len(split.test_indices),
        config_hash=_config_hash(config),
        metrics=report,
        notes=notes,
    )


def _config_hash(config: ExperimentConfig) -> str:
    """SHA-256 over a canonical JSON serialisation of ``config``."""
    payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_finetune_notes(
    *, notes: str, trainer: TrainerPort, training_result: TrainingResult
) -> str:
    """Append ``training_result`` summary to the model card notes.

    Per FINE_TUNING_DESIGN.md §3.2 the training bookkeeping is recorded in
    the model card's ``notes`` field. We append a one-line summary; the
    full result is otherwise reachable via the trainer's checkpoint dir
    (when configured).
    """
    suffix = (
        f"\n[finetune] trainer={trainer.identifier} "
        f"n_epochs_run={training_result.n_epochs_run} "
        f"best_epoch={training_result.best_epoch} "
        f"final_checkpoint={training_result.final_checkpoint_uri or 'none'}"
    )
    return notes + suffix
