"""Classifier head port: features -> per-class probabilities (multi-label).

Multi-label means each label is independent: probabilities are produced by a
per-class sigmoid, never a softmax over labels. The port is shape-only;
adapters are free to wrap sklearn / torch / whatever underneath.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class ClassifierHeadPort(Protocol):
    """Multi-label classifier head trained on extracted features."""

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int8],
    ) -> None:
        """Fit the head on ``(N, D)`` features and ``(N, K)`` multi-hot labels.

        Implementations must be deterministic given the same ``(features,
        labels)`` pair: two heads constructed identically and fit on identical
        data must produce identical ``predict_proba`` outputs.
        """
        ...

    def predict_proba(
        self,
        features: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Return per-class probabilities of shape ``(N, K)``.

        Calling ``predict_proba`` before :meth:`fit` must raise
        :class:`harness.domain.errors.ContractViolation`. All output values
        must lie in ``[0, 1]``.
        """
        ...
