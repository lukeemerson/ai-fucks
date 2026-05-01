"""Real per-class calibrator adapters built on scikit-learn.

Two implementations of :class:`harness.ports.calibrator.CalibratorPort`:

* :class:`PerClassPlattCalibrator` -- one
  :class:`sklearn.linear_model.LogisticRegression` (sigmoid) per class, fitted
  on ``(probs[:, k:k+1], labels[:, k])``. Per the architecture spec this is the
  preferred default for small positive counts (parametric, low variance).
* :class:`PerClassIsotonicCalibrator` -- one
  :class:`sklearn.isotonic.IsotonicRegression` per class with
  ``out_of_bounds='clip'``. Non-parametric and monotone by construction; better
  when sample counts per class are sufficiently large.

Both classes operate on each label column independently -- no information ever
crosses columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from harness.domain.errors import ContractViolation

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _validate_fit_inputs(
    probs: NDArray[np.float32], labels: NDArray[np.int8]
) -> None:
    if probs.ndim != 2:
        raise ContractViolation(f"probs must be 2-D, got ndim={probs.ndim}")
    if labels.ndim != 2:
        raise ContractViolation(f"labels must be 2-D, got ndim={labels.ndim}")
    if probs.shape != labels.shape:
        raise ContractViolation(
            f"shape mismatch: probs={probs.shape} labels={labels.shape}"
        )


def _validate_transform_input(probs: NDArray[np.float32]) -> None:
    if probs.ndim != 2:
        raise ContractViolation(f"probs must be 2-D, got ndim={probs.ndim}")


# ---------------------------------------------------------------------------
# Platt (sigmoid) calibrator
# ---------------------------------------------------------------------------


class PerClassPlattCalibrator:
    """Per-class sigmoid calibration via :class:`LogisticRegression`.

    For each class ``k`` an ``lbfgs``-solved logistic regression with ``C=1.0``
    is fitted on the single feature ``probs[:, k]`` against the binary label
    column ``labels[:, k]``. ``transform`` returns ``predict_proba(...)[:, 1]``
    per class.

    Degenerate single-class columns (all positives or all negatives) are
    handled by emitting a constant probability equal to the empirical mean of
    that column's labels -- avoiding the sklearn "needs >= 2 classes" error
    while keeping the output in ``[0, 1]``.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._models: tuple[LogisticRegression | None, ...] = ()
        # Constants for degenerate (single-class) columns.
        self._constants: tuple[float, ...] = ()
        self._n_classes: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        probs: NDArray[np.float32],
        labels: NDArray[np.int8],
    ) -> None:
        _validate_fit_inputs(probs, labels)
        n_classes = probs.shape[1]
        models: list[LogisticRegression | None] = []
        constants: list[float] = []
        for k in range(n_classes):
            x_k = probs[:, k : k + 1].astype(np.float64, copy=False)
            y_k = labels[:, k].astype(np.int64, copy=False)
            unique = np.unique(y_k)
            if unique.size < 2:
                # Degenerate column: store empirical mean (0.0 or 1.0).
                models.append(None)
                constants.append(float(y_k.mean()) if y_k.size > 0 else 0.0)
                continue
            model = LogisticRegression(solver="lbfgs", C=1.0)
            model.fit(x_k, y_k)
            models.append(model)
            constants.append(float("nan"))
        self._models = tuple(models)
        self._constants = tuple(constants)
        self._n_classes = n_classes
        self._fitted = True

    def transform(self, probs: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self._fitted:
            raise ContractViolation(
                "transform called before fit on PerClassPlattCalibrator"
            )
        _validate_transform_input(probs)
        if probs.shape[1] != self._n_classes:
            raise ContractViolation(
                f"transform expected {self._n_classes} columns, "
                f"got {probs.shape[1]}"
            )
        n_rows = probs.shape[0]
        out = np.empty((n_rows, self._n_classes), dtype=np.float32)
        for k in range(self._n_classes):
            model = self._models[k]
            if model is None:
                out[:, k] = np.float32(self._constants[k])
                continue
            x_k = probs[:, k : k + 1].astype(np.float64, copy=False)
            preds = model.predict_proba(x_k)[:, 1]
            out[:, k] = np.clip(preds, 0.0, 1.0).astype(np.float32, copy=False)
        return out


# ---------------------------------------------------------------------------
# Isotonic calibrator
# ---------------------------------------------------------------------------


class PerClassIsotonicCalibrator:
    """Per-class isotonic calibration via :class:`IsotonicRegression`.

    For each class ``k`` a non-parametric monotone regressor is fitted on
    ``probs[:, k]`` against ``labels[:, k]`` with ``out_of_bounds='clip'``.
    ``transform`` evaluates the regressor pointwise and clips the result to
    ``[0, 1]`` (isotonic regression on 0/1 targets is already in range, but
    NaNs from degenerate fits are guarded explicitly).

    Degenerate single-class columns emit a constant equal to the empirical
    mean of that column's labels.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._models: tuple[IsotonicRegression | None, ...] = ()
        self._constants: tuple[float, ...] = ()
        self._n_classes: int = 0

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        probs: NDArray[np.float32],
        labels: NDArray[np.int8],
    ) -> None:
        _validate_fit_inputs(probs, labels)
        n_classes = probs.shape[1]
        models: list[IsotonicRegression | None] = []
        constants: list[float] = []
        for k in range(n_classes):
            x_k = probs[:, k].astype(np.float64, copy=False)
            y_k = labels[:, k].astype(np.float64, copy=False)
            unique = np.unique(y_k)
            if unique.size < 2:
                models.append(None)
                constants.append(float(y_k.mean()) if y_k.size > 0 else 0.0)
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(x_k, y_k)
            models.append(model)
            constants.append(float("nan"))
        self._models = tuple(models)
        self._constants = tuple(constants)
        self._n_classes = n_classes
        self._fitted = True

    def transform(self, probs: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self._fitted:
            raise ContractViolation(
                "transform called before fit on PerClassIsotonicCalibrator"
            )
        _validate_transform_input(probs)
        if probs.shape[1] != self._n_classes:
            raise ContractViolation(
                f"transform expected {self._n_classes} columns, "
                f"got {probs.shape[1]}"
            )
        n_rows = probs.shape[0]
        out = np.empty((n_rows, self._n_classes), dtype=np.float32)
        for k in range(self._n_classes):
            model = self._models[k]
            if model is None:
                out[:, k] = np.float32(self._constants[k])
                continue
            x_k = probs[:, k].astype(np.float64, copy=False)
            preds = model.predict(x_k)
            out[:, k] = np.clip(preds, 0.0, 1.0).astype(np.float32, copy=False)
        return out
