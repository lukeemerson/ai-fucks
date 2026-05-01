"""Public surface of the harness domain layer.

Re-exports every dataclass from :mod:`harness.domain.types` and every error
from :mod:`harness.domain.errors` so callers can do
``from harness.domain import Sample`` without reaching into submodules.
"""

from __future__ import annotations

from harness.domain.errors import (
    AdapterError,
    ConfigError,
    ContractViolation,
    DataError,
    HarnessError,
)
from harness.domain.types import (
    BootstrapConfig,
    Dataset,
    ExperimentConfig,
    ExperimentResult,
    MetricInterval,
    MetricReport,
    ModelCard,
    PerClassMetric,
    Predictions,
    Probabilities,
    Sample,
    Split,
    ThresholdConfig,
    ThresholdSet,
)

__all__ = [
    "AdapterError",
    "BootstrapConfig",
    "ConfigError",
    "ContractViolation",
    "DataError",
    "Dataset",
    "ExperimentConfig",
    "ExperimentResult",
    "HarnessError",
    "MetricInterval",
    "MetricReport",
    "ModelCard",
    "PerClassMetric",
    "Predictions",
    "Probabilities",
    "Sample",
    "Split",
    "ThresholdConfig",
    "ThresholdSet",
]
