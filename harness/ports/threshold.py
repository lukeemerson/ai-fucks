"""ThresholdPort -- per-class operating-threshold tuning and application.

See ARCHITECTURE.md section 4.6. The port models two responsibilities:

* :meth:`ThresholdPort.fit` derives one operating threshold per class from
  out-of-fold calibrated probabilities and the matching multi-hot labels,
  honouring the :class:`~harness.domain.types.ThresholdConfig` (method,
  shrinkage, clamps).
* :meth:`ThresholdPort.apply` materialises binary
  :class:`~harness.domain.types.Predictions` from a probability matrix using
  a previously fitted :class:`~harness.domain.types.ThresholdSet`.

Adapters are stateless: every call is a pure function of its inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from harness.domain.types import (
    Predictions,
    Probabilities,
    ThresholdConfig,
    ThresholdSet,
)


@runtime_checkable
class ThresholdPort(Protocol):
    """Per-class operating-threshold port."""

    @property
    def identifier(self) -> str:
        """Stable identifier for this adapter (used in model cards)."""
        ...

    def fit(
        self,
        calibrated_oof: Probabilities,
        labels: Sequence[Sequence[int]],
        *,
        config: ThresholdConfig,
    ) -> ThresholdSet:
        """Return a per-class :class:`ThresholdSet` fitted on OOF probs."""
        ...

    def apply(
        self,
        probabilities: Probabilities,
        thresholds: ThresholdSet,
    ) -> Predictions:
        """Apply per-class thresholds to ``probabilities`` element-wise."""
        ...
