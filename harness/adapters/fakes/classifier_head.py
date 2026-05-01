"""Linear fake classifier head: tiny per-class logistic regression in numpy.

This adapter exists so contract tests can exercise the
:class:`harness.ports.classifier_head.ClassifierHeadPort` shape without
importing sklearn or torch. The fitting routine is a single closed-form
ridge-regression step per label on sigmoid pseudo-labels -- enough to make
``predict_proba`` deterministic and shape-correct, which is all the contract
demands.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from harness.domain.errors import ContractViolation


def _sigmoid(z: NDArray[np.float32]) -> NDArray[np.float32]:
    # Clip for numerical stability; sigmoid saturates outside ~|30|.
    z_clipped = np.clip(z, -30.0, 30.0).astype(np.float32, copy=False)
    return (1.0 / (1.0 + np.exp(-z_clipped))).astype(np.float32, copy=False)


class LinearFakeClassifierHead:
    """Multi-label linear head fit by ridge regression on +/-1 targets."""

    def __init__(self, n_labels: int, ridge: float = 1.0) -> None:
        if n_labels <= 0:
            raise ContractViolation(f"n_labels must be positive, got {n_labels}")
        if ridge <= 0.0:
            raise ContractViolation(f"ridge must be positive, got {ridge}")
        self._n_labels: int = n_labels
        self._ridge: float = float(ridge)
        self._weights: NDArray[np.float32] | None = None
        self._bias: NDArray[np.float32] | None = None

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int8],
    ) -> None:
        if features.ndim != 2:
            raise ContractViolation(
                f"features must be 2-D, got ndim={features.ndim}"
            )
        if labels.ndim != 2:
            raise ContractViolation(
                f"labels must be 2-D, got ndim={labels.ndim}"
            )
        if labels.shape[0] != features.shape[0]:
            raise ContractViolation(
                f"row mismatch: features={features.shape[0]} labels={labels.shape[0]}"
            )
        if labels.shape[1] != self._n_labels:
            raise ContractViolation(
                f"expected {self._n_labels} labels, got {labels.shape[1]}"
            )
        x = features.astype(np.float32, copy=False)
        # +/-1 targets so the closed-form least-squares solution gives a usable
        # linear logit. Cast to float32 for downstream determinism.
        y = (2.0 * labels.astype(np.float32) - 1.0).astype(np.float32, copy=False)
        d = int(x.shape[1])
        # Ridge: w = (X^T X + lambda I)^-1 X^T y, with column-mean centering as
        # the bias estimate.
        x_mean = x.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
        x_centered = (x - x_mean).astype(np.float32, copy=False)
        gram = x_centered.T @ x_centered + self._ridge * np.eye(d, dtype=np.float32)
        rhs = x_centered.T @ y
        weights = np.linalg.solve(gram, rhs).astype(np.float32, copy=False)
        # Bias is mean target minus mean-feature contribution (which is zero by
        # construction since features are centered).
        bias = y.mean(axis=0).astype(np.float32, copy=False)
        self._weights = weights
        self._bias = bias
        # Stash the centering vector so predict can mirror the fit transform.
        self._x_mean: NDArray[np.float32] = x_mean

    def predict_proba(
        self,
        features: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        if self._weights is None or self._bias is None:
            raise ContractViolation(
                "predict_proba called before fit on LinearFakeClassifierHead"
            )
        if features.ndim != 2:
            raise ContractViolation(
                f"features must be 2-D, got ndim={features.ndim}"
            )
        x = features.astype(np.float32, copy=False)
        x_centered = (x - self._x_mean).astype(np.float32, copy=False)
        logits = (x_centered @ self._weights + self._bias).astype(
            np.float32, copy=False
        )
        return _sigmoid(logits)
