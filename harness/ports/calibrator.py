"""Calibrator port: per-class probability calibration.

Per-class means each label column is calibrated independently of the others;
adapters must not mix information across columns. ``fit`` is called once on
validation OOF probabilities; ``transform`` is then applied to validation and
test probability matrices.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class CalibratorPort(Protocol):
    """Per-class probability calibrator."""

    @property
    def is_fitted(self) -> bool:
        """``True`` once :meth:`fit` has been called, ``False`` before."""
        ...

    def fit(
        self,
        probs: NDArray[np.float32],
        labels: NDArray[np.int8],
    ) -> None:
        """Fit calibration parameters per class.

        ``probs`` is ``(N, K)`` raw probabilities; ``labels`` is the matching
        ``(N, K)`` multi-hot label matrix. Implementations must operate on
        each column independently.
        """
        ...

    def transform(self, probs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return calibrated probabilities of the same shape as ``probs``.

        Calling ``transform`` before :meth:`fit` must raise
        :class:`harness.domain.errors.ContractViolation`. All output values
        must lie in ``[0, 1]``.
        """
        ...
