"""Composition root for harness experiments.

``run_experiment`` is the only function that wires every port together. All
adapters are passed in as keyword arguments; this module never imports an
adapter directly. The companion :mod:`harness.composition.factories` module
constructs concrete adapter sets.

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

from harness.domain.errors import ContractViolation
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

__all__ = ["run_experiment", "BYTES_IMAGE_SHAPE"]

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
    """
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
        notes=config.notes,
    )


def _config_hash(config: ExperimentConfig) -> str:
    """SHA-256 over a canonical JSON serialisation of ``config``."""
    payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
