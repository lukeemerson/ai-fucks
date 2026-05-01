"""MetricsPort -- per-class and macro evaluation with bootstrap CIs.

See ARCHITECTURE.md section 4.7. The port computes per-class precision /
recall / F1 / AUROC / AUPRC together with their macro aggregates and a
nonparametric bootstrap confidence interval for each, all from a probability
matrix, multi-hot labels, and a fitted :class:`ThresholdSet`.

Adapters must be deterministic functions of ``(inputs, bootstrap.seed)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from harness.domain.types import (
    BootstrapConfig,
    MetricReport,
    Probabilities,
    ThresholdSet,
)


@runtime_checkable
class MetricsPort(Protocol):
    """Multi-label evaluation port."""

    def evaluate(
        self,
        probabilities: Probabilities,
        labels: Sequence[Sequence[int]],
        thresholds: ThresholdSet,
        *,
        bootstrap: BootstrapConfig,
    ) -> MetricReport:
        """Return a :class:`MetricReport` with macro and per-class metrics."""
        ...
