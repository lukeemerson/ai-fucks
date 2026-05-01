"""Domain dataclasses for the harness.

This module is the pure types layer. Per ARCHITECTURE.md section 3 every type
here is ``@dataclass(frozen=True, slots=True)`` and does no I/O.

Dependency exception
--------------------
The architecture spec calls for stdlib-only types in ``domain/``. We deviate
on a single dependency: ``numpy``. Probability and prediction matrices are
modelled as ``numpy.typing.NDArray[...]`` because (a) numpy is already a hard
project dependency, (b) it is the lingua franca every adapter
(sklearn/torch/fs) speaks at its boundary, and (c) it lets us validate
shape/range invariants vectorially in ``__post_init__``. No other third-party
imports are permitted in this module. ``sklearn``, ``torch``, and any
``harness.adapters`` / ``harness.ports`` / ``harness.composition`` import is
forbidden.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from harness.domain.errors import ConfigError, ContractViolation

# ---------------------------------------------------------------------------
# Sample / Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Sample:
    """A single image in a dataset, with its multi-hot label vector."""

    sample_id: str
    patient_id: str
    image_ref: str
    labels: tuple[int, ...]
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Dataset:
    """A named collection of samples sharing a label vocabulary."""

    name: str
    label_names: tuple[str, ...]
    samples: tuple[Sample, ...]

    def __post_init__(self) -> None:
        n_labels = len(self.label_names)
        for sample in self.samples:
            if len(sample.labels) != n_labels:
                raise ContractViolation(
                    f"sample {sample.sample_id!r}: labels length "
                    f"{len(sample.labels)} != len(label_names) {n_labels}"
                )


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Split:
    """Train / val / test index assignment produced by a SplitterPort."""

    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ContractViolation(f"seed must be non-negative, got {self.seed}")
        for name, indices in (
            ("train_indices", self.train_indices),
            ("val_indices", self.val_indices),
            ("test_indices", self.test_indices),
        ):
            for idx in indices:
                if idx < 0:
                    raise ContractViolation(f"{name} contains negative index {idx}")
        train = set(self.train_indices)
        val = set(self.val_indices)
        test = set(self.test_indices)
        if train & val or train & test or val & test:
            raise ContractViolation("train/val/test indices must be disjoint")


# ---------------------------------------------------------------------------
# Probabilities / Predictions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Probabilities:
    """Per-sample, per-class probabilities in ``[0, 1]``."""

    sample_ids: tuple[str, ...]
    label_names: tuple[str, ...]
    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ContractViolation(
                f"values must be 2-D, got ndim={self.values.ndim}"
            )
        n_rows, n_cols = self.values.shape
        if n_rows != len(self.sample_ids):
            raise ContractViolation(
                f"row count {n_rows} != len(sample_ids) {len(self.sample_ids)}"
            )
        if n_cols != len(self.label_names):
            raise ContractViolation(
                f"column count {n_cols} != len(label_names) {len(self.label_names)}"
            )
        if self.values.size > 0:
            vmin = float(self.values.min())
            vmax = float(self.values.max())
            if vmin < 0.0 or vmax > 1.0:
                raise ContractViolation(
                    f"values must be in [0, 1], got min={vmin}, max={vmax}"
                )


@dataclass(frozen=True, slots=True)
class Predictions:
    """Per-sample, per-class binary predictions (0/1)."""

    sample_ids: tuple[str, ...]
    label_names: tuple[str, ...]
    values: NDArray[np.int8]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ContractViolation(
                f"values must be 2-D, got ndim={self.values.ndim}"
            )
        n_rows, n_cols = self.values.shape
        if n_rows != len(self.sample_ids):
            raise ContractViolation(
                f"row count {n_rows} != len(sample_ids) {len(self.sample_ids)}"
            )
        if n_cols != len(self.label_names):
            raise ContractViolation(
                f"column count {n_cols} != len(label_names) {len(self.label_names)}"
            )
        if self.values.size > 0:
            unique = np.unique(self.values)
            for v in unique.tolist():
                if v not in (0, 1):
                    raise ContractViolation(
                        f"values must be 0 or 1, got {v}"
                    )


# ---------------------------------------------------------------------------
# Metric report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricInterval:
    """Point estimate plus lower/upper confidence bounds.

    The only enforced invariant is ``lower <= upper``. The ``point`` estimate
    is permitted to fall outside ``[lower, upper]`` because raw bootstrap
    quantiles can legitimately fail to bracket the full-sample point estimate
    on degenerate or tiny resample sets. We report the raw bootstrap bounds
    rather than silently widening them.
    """

    point: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ContractViolation(
                f"lower {self.lower} must be <= upper {self.upper}"
            )


@dataclass(frozen=True, slots=True)
class PerClassMetric:
    """F1 / AUROC / AUPRC (with CIs) and support for one label."""

    label: str
    f1: MetricInterval
    auroc: MetricInterval
    auprc: MetricInterval
    support: int

    def __post_init__(self) -> None:
        if self.support < 0:
            raise ContractViolation(
                f"support must be non-negative, got {self.support}"
            )


@dataclass(frozen=True, slots=True)
class MetricReport:
    """Macro and per-class metrics with bootstrap CIs."""

    macro_f1: MetricInterval
    macro_auroc: MetricInterval
    macro_auprc: MetricInterval
    per_class: tuple[PerClassMetric, ...]
    n_bootstrap: int
    seed: int

    def __post_init__(self) -> None:
        if not self.per_class:
            raise ContractViolation("per_class must be non-empty")
        if self.n_bootstrap <= 0:
            raise ContractViolation(
                f"n_bootstrap must be positive, got {self.n_bootstrap}"
            )
        if self.seed < 0:
            raise ContractViolation(f"seed must be non-negative, got {self.seed}")

    @property
    def macro_f1_mean(self) -> float:
        """Arithmetic mean of per-class F1 point estimates."""
        return sum(c.f1.point for c in self.per_class) / len(self.per_class)


# ---------------------------------------------------------------------------
# ThresholdSet
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThresholdSet:
    """Per-class operating thresholds plus the method that produced them."""

    label_names: tuple[str, ...]
    thresholds: tuple[float, ...]
    method: str
    shrinkage: float
    clamp_lo: float
    clamp_hi: float

    def __post_init__(self) -> None:
        if len(self.thresholds) != len(self.label_names):
            raise ContractViolation(
                f"len(thresholds) {len(self.thresholds)} != "
                f"len(label_names) {len(self.label_names)}"
            )
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ContractViolation(
                f"shrinkage must be in [0, 1], got {self.shrinkage}"
            )
        if not 0.0 <= self.clamp_lo <= 1.0:
            raise ContractViolation(
                f"clamp_lo must be in [0, 1], got {self.clamp_lo}"
            )
        if not 0.0 <= self.clamp_hi <= 1.0:
            raise ContractViolation(
                f"clamp_hi must be in [0, 1], got {self.clamp_hi}"
            )
        if self.clamp_lo > self.clamp_hi:
            raise ContractViolation(
                f"clamp_lo {self.clamp_lo} must be <= clamp_hi {self.clamp_hi}"
            )
        for label, t in zip(self.label_names, self.thresholds, strict=True):
            if not self.clamp_lo <= t <= self.clamp_hi:
                raise ContractViolation(
                    f"threshold for {label!r} = {t} outside "
                    f"[{self.clamp_lo}, {self.clamp_hi}]"
                )

    def threshold_for(self, label: str) -> float:
        """Return the threshold for ``label`` or raise ``KeyError``."""
        for name, t in zip(self.label_names, self.thresholds, strict=True):
            if name == label:
                return t
        raise KeyError(label)

    def __iter__(self) -> Iterator[tuple[str, float]]:
        return iter(zip(self.label_names, self.thresholds, strict=True))


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Persisted summary of a trained model and its evaluation."""

    name: str
    version: str
    created_at: datetime
    backbone: str
    head: str
    calibrator: str
    threshold_method: str
    label_names: tuple[str, ...]
    train_size: int
    val_size: int
    test_size: int
    config_hash: str
    metrics: MetricReport
    notes: str


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Bootstrap CI configuration for the metrics adapter."""

    n_resamples: int
    confidence: float
    seed: int

    def __post_init__(self) -> None:
        if self.n_resamples <= 0:
            raise ConfigError(
                f"n_resamples must be positive, got {self.n_resamples}"
            )
        # Open interval: bootstrap CI math degenerates at confidence==0 or 1.
        if not 0.0 < self.confidence < 1.0:
            raise ConfigError(
                f"confidence must be in (0, 1), got {self.confidence}"
            )
        if self.seed < 0:
            raise ConfigError(f"seed must be non-negative, got {self.seed}")


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """Per-class threshold-tuning configuration."""

    method: str
    shrinkage: float
    clamp_lo: float
    clamp_hi: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ConfigError(
                f"shrinkage must be in [0, 1], got {self.shrinkage}"
            )
        if not 0.0 <= self.clamp_lo <= 1.0:
            raise ConfigError(
                f"clamp_lo must be in [0, 1], got {self.clamp_lo}"
            )
        if not 0.0 <= self.clamp_hi <= 1.0:
            raise ConfigError(
                f"clamp_hi must be in [0, 1], got {self.clamp_hi}"
            )
        if self.clamp_lo > self.clamp_hi:
            raise ConfigError(
                f"clamp_lo {self.clamp_lo} must be <= clamp_hi {self.clamp_hi}"
            )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level configuration for a single experiment run."""

    experiment_name: str
    dataset_name: str
    label_names: tuple[str, ...]
    val_fraction: float
    test_fraction: float
    seed: int
    bootstrap: BootstrapConfig
    threshold: ThresholdConfig
    backbone_id: str
    head_id: str
    calibrator_id: str
    artifact_root: str
    notes: str

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ConfigError("experiment_name must be non-empty")
        if not self.label_names:
            raise ConfigError("label_names must be non-empty")
        if self.seed < 0:
            raise ConfigError(f"seed must be non-negative, got {self.seed}")
        if self.val_fraction < 0.0 or self.test_fraction < 0.0:
            raise ConfigError(
                "val_fraction and test_fraction must be non-negative, "
                f"got {self.val_fraction}, {self.test_fraction}"
            )
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ConfigError(
                "val_fraction + test_fraction must be < 1, "
                f"got {self.val_fraction + self.test_fraction}"
            )


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Everything produced by one full ``run_experiment`` call."""

    config: ExperimentConfig
    split: Split
    thresholds: ThresholdSet
    val_probabilities: Probabilities
    test_probabilities: Probabilities
    test_predictions: Predictions
    report: MetricReport
    model_card: ModelCard
    artifact_uris: Mapping[str, str]
