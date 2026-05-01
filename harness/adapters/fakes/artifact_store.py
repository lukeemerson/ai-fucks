"""``InMemoryFakeArtifactStore`` -- dict-backed ``ArtifactStorePort`` adapter.

Behavior
--------

* Every ``write_*`` method assigns the artifact a synthetic
  ``memory://`` path/URI keyed by the artifact kind (and, where relevant, the
  caller-supplied ``name``).
* Re-writing the same logical name overwrites the stored object and returns
  the *same* path. This idempotency is asserted by the contract suite.
* :meth:`get` is the inspection surface used by tests to retrieve what was
  written. It raises ``KeyError`` for unknown paths.

The adapter has no third-party dependencies beyond numpy (none used here).
It is safe to instantiate per-test.
"""

from __future__ import annotations

from harness.domain.types import (
    MetricReport,
    ModelCard,
    Predictions,
    ThresholdSet,
)

# A union over the artifact kinds this adapter accepts. We keep this as a
# concrete tuple type alias rather than ``Any`` to satisfy mypy --strict.
Artifact = ModelCard | Predictions | ThresholdSet | MetricReport


class InMemoryFakeArtifactStore:
    """Dict-backed artifact store with a ``memory://`` URI scheme."""

    def __init__(self) -> None:
        self._store: dict[str, Artifact] = {}

    # --- writes ------------------------------------------------------------

    def write_model_card(self, card: ModelCard) -> str:
        path = "memory://model_card.json"
        self._store[path] = card
        return path

    def write_predictions(self, preds: Predictions, name: str) -> str:
        path = f"memory://predictions/{name}.npy"
        self._store[path] = preds
        return path

    def write_thresholds(self, thresholds: ThresholdSet) -> str:
        path = "memory://thresholds.json"
        self._store[path] = thresholds
        return path

    def write_metric_report(self, report: MetricReport) -> str:
        path = "memory://metric_report.json"
        self._store[path] = report
        return path

    # --- inspection --------------------------------------------------------

    def get(self, path: str) -> Artifact:
        """Retrieve a previously written artifact or raise ``KeyError``."""
        if path not in self._store:
            raise KeyError(path)
        return self._store[path]

    def paths(self) -> tuple[str, ...]:
        """Return all written paths in insertion order."""
        return tuple(self._store.keys())
