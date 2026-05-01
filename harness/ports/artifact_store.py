"""ArtifactStorePort -- persistence surface for run artifacts.

This module defines the four required write methods owned by this slice of
the harness (model card, predictions, thresholds, metric report). Each method
returns a string path/URI; consumers never receive raw bytes back. The fs
adapter and the in-memory fake satisfy this protocol.

Per ARCHITECTURE.md section 4.8 the broader on-disk store may add additional
methods (data card, probabilities, weights, ``read_blob``); those live in a
separate slice and are not part of this minimum contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.domain.types import (
    MetricReport,
    ModelCard,
    Predictions,
    ThresholdSet,
)


@runtime_checkable
class ArtifactStorePort(Protocol):
    """Minimum write surface for run artifacts."""

    def write_model_card(self, card: ModelCard) -> str:
        """Persist ``card`` and return its path/URI."""
        ...

    def write_predictions(self, preds: Predictions, name: str) -> str:
        """Persist ``preds`` under logical ``name`` and return its path/URI."""
        ...

    def write_thresholds(self, thresholds: ThresholdSet) -> str:
        """Persist ``thresholds`` and return their path/URI."""
        ...

    def write_metric_report(self, report: MetricReport) -> str:
        """Persist ``report`` and return its path/URI."""
        ...
