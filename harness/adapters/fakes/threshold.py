"""``FixedFakeThreshold`` -- trivial deterministic ``ThresholdPort`` adapter.

Behavior
--------

* :meth:`fit` ignores the inputs (other than the label vocabulary and the
  config's clamp bounds) and returns a :class:`ThresholdSet` whose every
  threshold is the constant ``0.5`` (clamped into ``[clamp_lo, clamp_hi]``
  if the configured range excludes the midpoint).
* :meth:`apply` performs ``probs >= threshold`` element-wise, per class.

The adapter is intentionally minimal -- production threshold tuning lives
in ``adapters/sklearn/threshold.py``. This fake exists so contract tests
and integration tests can run without sklearn.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from harness.domain.errors import AdapterError, ContractViolation
from harness.domain.types import (
    Predictions,
    Probabilities,
    ThresholdConfig,
    ThresholdSet,
)


class FixedFakeThreshold:
    """Returns a constant 0.5 threshold per class; element-wise apply."""

    def __init__(self, value: float = 0.5) -> None:
        if not 0.0 <= value <= 1.0:
            # AdapterError, not ContractViolation: invalid init argument is a
            # mis-configuration, not a runtime port-contract violation.
            raise AdapterError(
                f"value must be in [0, 1], got {value}"
            )
        self._value = value

    @property
    def identifier(self) -> str:
        return f"fake-fixed@{self._value}"

    def fit(
        self,
        calibrated_oof: Probabilities,
        labels: Sequence[Sequence[int]],  # noqa: ARG002 -- fake ignores labels by design
        *,
        config: ThresholdConfig,
    ) -> ThresholdSet:
        # Clamp the constant into the configured range so the resulting
        # ThresholdSet always honours its invariants.
        clamped = min(max(self._value, config.clamp_lo), config.clamp_hi)
        thresholds = tuple(clamped for _ in calibrated_oof.label_names)
        return ThresholdSet(
            label_names=calibrated_oof.label_names,
            thresholds=thresholds,
            method=config.method or "fake-fixed",
            shrinkage=config.shrinkage,
            clamp_lo=config.clamp_lo,
            clamp_hi=config.clamp_hi,
        )

    def apply(
        self,
        probabilities: Probabilities,
        thresholds: ThresholdSet,
    ) -> Predictions:
        if thresholds.label_names != probabilities.label_names:
            raise ContractViolation(
                "thresholds.label_names "
                f"{thresholds.label_names!r} != probabilities.label_names "
                f"{probabilities.label_names!r}"
            )
        thr = np.asarray(thresholds.thresholds, dtype=np.float32)
        # Broadcast over rows: (n_samples, n_labels) >= (n_labels,).
        binary = (probabilities.values >= thr[None, :]).astype(np.int8)
        return Predictions(
            sample_ids=probabilities.sample_ids,
            label_names=probabilities.label_names,
            values=binary,
        )
