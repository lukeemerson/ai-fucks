"""Public surface of the ``harness`` package.

Per ARCHITECTURE.md section 8.8 (and §13.2 of the v1-implemented surface
addendum), the public API is intentionally small:

* The composition root: :func:`run_experiment`.
* The v1 factories that wire concrete adapter sets:
  :func:`build_v1_runner_with_fakes`, :func:`build_v1_runner_sklearn`.
* Every domain type re-exported from :mod:`harness.domain`.

Ports and adapter classes are *not* part of this surface. Consumers should
construct adapter sets through the factories rather than importing
``harness.adapters.*`` directly.
"""

from __future__ import annotations

from harness.composition.factories import (
    RunnerBundle,
    build_v1_runner_sklearn,
    build_v1_runner_with_fakes,
)
from harness.composition.runner import run_experiment
from harness.domain import (
    AdapterError,
    BootstrapConfig,
    ConfigError,
    ContractViolation,
    DataError,
    Dataset,
    ExperimentConfig,
    ExperimentResult,
    HarnessError,
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
    "RunnerBundle",
    "Sample",
    "Split",
    "ThresholdConfig",
    "ThresholdSet",
    "build_v1_runner_sklearn",
    "build_v1_runner_with_fakes",
    "run_experiment",
]
