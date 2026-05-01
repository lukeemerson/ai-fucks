"""``SklearnGradientBoostingHead`` -- production ``ClassifierHeadPort`` adapter.

One :class:`sklearn.ensemble.HistGradientBoostingClassifier` per output class
(multi-label means each label is independent; per-class sigmoid, never a
softmax over labels). Each per-class classifier is seeded deterministically
from the constructor seed so two heads built with the same ``(seed, data)``
pair produce byte-identical ``predict_proba`` output.

Hyperparameters
---------------
``max_iter`` defaults to ``200`` (boosting rounds; sklearn default is 100,
we want a little more headroom for tabular features). The constructor
accepts an explicit ``max_iter`` keyword override for tests that don't need
real learning and care about wall-clock budget. ``early_stopping="auto"``
(sklearn default -- enables only when ``n_samples > 10_000`` so it never
triggers a stratified split on the small CV folds we use for v1).
Production-tuning of ``max_iter`` is deferred to the v1.1 ablations (Step 4
of ``PAPER_CHECKLIST.md``).

Degenerate columns
------------------
``HistGradientBoostingClassifier.fit`` requires at least two distinct classes.
When a column has all-zero or all-one labels in the training fold (which is
common for rare findings on tiny CV folds) we fall back to emitting a constant
probability equal to the column's empirical positive rate (``0.0`` or
``1.0``). This keeps the output well-defined and inside ``[0, 1]`` without
crashing the pipeline.

Seed range
----------
sklearn's ``random_state`` parameter requires an int in ``[0, 2**32)``. The
harness ``RandomnessPort.child_seed`` returns 63-bit values, so we mask the
incoming seed to 32 bits before constructing each per-class estimator.
Determinism is preserved: two calls with the same input ``seed`` derive the
same masked ``(seed + k) & 0xFFFFFFFF`` value for every column ``k``.

Per-class seed derivation
-------------------------
Each of the ``K`` :class:`HistGradientBoostingClassifier` instances uses
``random_state = (seed + k) & 0xFFFFFFFF`` (the mask enforces sklearn's
32-bit cap). This is collision-free for any plausible ``K`` (well under
``2**32``) and is intentionally **not** routed through
``RandomnessPort.child_seed``: the head adapter must remain port-free so its
constructor signature is dependency-injection-light. Determinism is still
absolute -- same ``(seed, K, fit-data)`` produces byte-identical
``predict_proba`` outputs across runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from harness.domain.errors import ContractViolation

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["SklearnGradientBoostingHead"]


_MAX_ITER: int = 200
# "auto" is the sklearn default; HistGBC enables early stopping only when
# n_samples > 10_000. This avoids the internal stratified validation split
# that fails on the tiny CV folds used by the v1 publication pipeline.
_EARLY_STOPPING: str = "auto"
# sklearn's ``random_state`` parameter validator caps at ``2**32 - 1``.
_RANDOM_STATE_MASK: int = 0xFFFFFFFF


class SklearnGradientBoostingHead:
    """Multi-label GBT classifier head: one HistGBC per output column.

    Determinism: the per-class :class:`HistGradientBoostingClassifier` for
    output column ``i`` is constructed with
    ``random_state = (seed + i) & 0xFFFFFFFF`` so that ``(seed, data)``
    uniquely determines ``predict_proba`` output. The mask only matters when
    the caller passes a seed wider than 32 bits (e.g. the harness's 63-bit
    ``child_seed`` outputs).

    Multi-label policy: ``K`` independent classifiers, no information ever
    crosses output columns. Per-class fallbacks for degenerate single-class
    folds emit the empirical positive rate (``0.0`` or ``1.0``) for every
    row in the column.
    """

    def __init__(
        self,
        n_labels: int,
        seed: int,
        *,
        max_iter: int = _MAX_ITER,
    ) -> None:
        if n_labels <= 0:
            raise ContractViolation(f"n_labels must be positive, got {n_labels}")
        if max_iter <= 0:
            raise ContractViolation(
                f"max_iter must be positive, got {max_iter}"
            )
        self._n_labels: int = int(n_labels)
        self._seed: int = int(seed)
        self._max_iter: int = int(max_iter)
        # ``None`` for a degenerate column; filled with the fitted estimator
        # otherwise. Parallel to ``_constants`` which holds the empirical
        # positive rate for degenerate columns and ``NaN`` otherwise.
        self._models: tuple[HistGradientBoostingClassifier | None, ...] = ()
        self._constants: tuple[float, ...] = ()
        self._fitted: bool = False

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
                f"row mismatch: features={features.shape[0]} "
                f"labels={labels.shape[0]}"
            )
        if labels.shape[1] != self._n_labels:
            raise ContractViolation(
                f"expected {self._n_labels} labels, got {labels.shape[1]}"
            )

        x = features.astype(np.float32, copy=False)
        models: list[HistGradientBoostingClassifier | None] = []
        constants: list[float] = []
        for k in range(self._n_labels):
            y_k = labels[:, k].astype(np.int64, copy=False)
            unique = np.unique(y_k)
            if unique.size < 2:
                # Degenerate column: HistGBC requires >= 2 classes. Fall back
                # to the empirical positive rate (0.0 or 1.0) -- which equals
                # the unique value present, so y_k.mean() is correct.
                models.append(None)
                constants.append(float(y_k.mean()) if y_k.size > 0 else 0.0)
                continue
            model = HistGradientBoostingClassifier(
                max_iter=self._max_iter,
                early_stopping=_EARLY_STOPPING,
                random_state=(self._seed + k) & _RANDOM_STATE_MASK,
            )
            model.fit(x, y_k)
            models.append(model)
            constants.append(float("nan"))

        self._models = tuple(models)
        self._constants = tuple(constants)
        self._fitted = True

    def predict_proba(
        self,
        features: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        if not self._fitted:
            raise ContractViolation(
                "predict_proba called before fit on SklearnGradientBoostingHead"
            )
        if features.ndim != 2:
            raise ContractViolation(
                f"features must be 2-D, got ndim={features.ndim}"
            )
        x = features.astype(np.float32, copy=False)
        n_rows = x.shape[0]
        out = np.empty((n_rows, self._n_labels), dtype=np.float32)
        for k in range(self._n_labels):
            model = self._models[k]
            if model is None:
                out[:, k] = np.float32(self._constants[k])
                continue
            # ``predict_proba`` returns columns in ``classes_`` order. With
            # binary labels and at-least-one positive sample this is always
            # ``[0, 1]``, so column 1 is P(y=1).
            preds = model.predict_proba(x)[:, 1]
            out[:, k] = np.clip(preds, 0.0, 1.0).astype(np.float32, copy=False)
        return out
