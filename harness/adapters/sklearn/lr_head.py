"""``SklearnLogisticRegressionHead`` -- production ``ClassifierHeadPort`` adapter.

One :class:`sklearn.linear_model.LogisticRegression` per output class
(multi-label means each label is independent; per-class sigmoid, never a
softmax over labels). Each per-class classifier is seeded deterministically
from the constructor seed so two heads built with the same ``(seed, data)``
pair produce byte-identical ``predict_proba`` output.

Why this head exists
--------------------
On TXRV-DenseNet121-NIH 1024-d embeddings, an ad-hoc ablation produced:

* ``SklearnGradientBoostingHead``: macro-F1 = 0.094, five all-zero classes
  (Mass, Consolidation, Edema, Fibrosis, Pleural_Thickening).
* ``SklearnLogisticRegressionHead`` (this adapter, defaults below):
  macro-F1 = 0.157 (+0.063), zero all-zero classes, ~zero AUROC cost.

The lift comes almost entirely from ``class_weight='balanced'``: the rare
classes have <1% prevalence in NIH-14, and an unweighted LR collapses to
"always negative" on those columns. The balanced weights re-weight each
class's contribution to the loss by ``n_samples / (n_classes * n_class_k)``,
which lets gradients from rare-class positives compete with the abundant
negatives. The same trick is unavailable to :class:`HistGradientBoostingClassifier`
in any equivalent form.

Hyperparameters
---------------
``class_weight``: ``"balanced"`` (default) or ``"none"``. The default is the
load-bearing rare-class rescue described above. ``"none"`` is exposed only so
that ablations can demonstrate the rescue effect; production runs should
leave it at ``"balanced"``.

``C``: L2 regularization strength (sklearn convention -- *smaller* is
stronger). Default ``1.0``. The ablation locked this; it is not tuned.

``max_iter``: lbfgs iteration cap. Default ``1000`` to suppress sklearn's
``ConvergenceWarning`` on 1024-d embeddings.

Solver
------
``solver="lbfgs"`` (sklearn default for L2 binary LR). The multi-label loop
is single-threaded by construction (the per-class fit itself owns whatever
internal parallelism sklearn provides); ``n_jobs`` is **not** passed because
sklearn 1.8 deprecated the keyword for :class:`LogisticRegression` and the
default behaviour matches our prior single-thread setting.

Degenerate columns
------------------
:class:`LogisticRegression.fit` requires at least two distinct classes.
When a column has all-zero or all-one labels in the training fold (which is
common for the rarest findings on tiny CV folds) we fall back to emitting
a constant probability equal to the column's empirical positive rate
(``0.0`` or ``1.0``). This keeps the output well-defined and inside
``[0, 1]`` without crashing the pipeline. This mirrors
:class:`SklearnGradientBoostingHead`'s policy verbatim.

Seed range
----------
sklearn's ``random_state`` parameter requires an int in ``[0, 2**32)``. The
harness ``RandomnessPort.child_seed`` returns 63-bit values, so we mask the
incoming seed to 32 bits before constructing each per-class estimator.
Determinism is preserved: two calls with the same input ``seed`` derive the
same masked ``(seed + k) & 0xFFFFFFFF`` value for every column ``k``.

Per-class seed derivation
-------------------------
Each of the ``K`` :class:`LogisticRegression` instances uses
``random_state = (seed + k) & 0xFFFFFFFF`` (the mask enforces sklearn's
32-bit cap). This is collision-free for any plausible ``K`` (well under
``2**32``) and is intentionally **not** routed through
``RandomnessPort.child_seed``: the head adapter must remain port-free so
its constructor signature is dependency-injection-light. Determinism is
still absolute -- same ``(seed, K, fit-data)`` produces byte-identical
``predict_proba`` outputs across runs. The lbfgs solver is itself
deterministic given a fixed ``random_state``, training data, and class
weights, so ``random_state`` mostly serves as a defence-in-depth signal
for any sklearn internals that consume it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from harness.domain.errors import ContractViolation

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["SklearnLogisticRegressionHead"]


# sklearn's ``random_state`` parameter validator caps at ``2**32 - 1``.
_RANDOM_STATE_MASK: int = 0xFFFFFFFF
_DEFAULT_MAX_ITER: int = 1000
_DEFAULT_C: float = 1.0
_SOLVER: str = "lbfgs"

ClassWeight = Literal["balanced", "none"]


class SklearnLogisticRegressionHead:
    """Multi-label LR classifier head: one binary LR per output column.

    Determinism: the per-class :class:`LogisticRegression` for output
    column ``i`` is constructed with
    ``random_state = (seed + i) & 0xFFFFFFFF`` so that ``(seed, data)``
    uniquely determines ``predict_proba`` output. The mask only matters
    when the caller passes a seed wider than 32 bits (e.g. the harness's
    63-bit ``child_seed`` outputs).

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
        class_weight: ClassWeight = "balanced",
        c_regularization: float = _DEFAULT_C,
        max_iter: int = _DEFAULT_MAX_ITER,
    ) -> None:
        if n_labels <= 0:
            raise ContractViolation(f"n_labels must be positive, got {n_labels}")
        if max_iter <= 0:
            raise ContractViolation(
                f"max_iter must be positive, got {max_iter}"
            )
        if c_regularization <= 0:
            raise ContractViolation(
                f"c_regularization must be positive, got {c_regularization}"
            )
        self._n_labels: int = int(n_labels)
        self._seed: int = int(seed)
        self._class_weight: ClassWeight = class_weight
        self._c: float = float(c_regularization)
        self._max_iter: int = int(max_iter)
        # ``None`` for a degenerate column; filled with the fitted estimator
        # otherwise. Parallel to ``_constants`` which holds the empirical
        # positive rate for degenerate columns and ``NaN`` otherwise.
        self._models: tuple[LogisticRegression | None, ...] = ()
        self._constants: tuple[float, ...] = ()
        self._fitted: bool = False

    def _sklearn_class_weight(self) -> str | None:
        """Translate the literal config into sklearn's expected value."""
        if self._class_weight == "balanced":
            return "balanced"
        return None

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
        sk_class_weight = self._sklearn_class_weight()
        models: list[LogisticRegression | None] = []
        constants: list[float] = []
        for k in range(self._n_labels):
            y_k = labels[:, k].astype(np.int64, copy=False)
            unique = np.unique(y_k)
            if unique.size < 2:
                # Degenerate column: LR requires >= 2 classes. Fall back to
                # the empirical positive rate (0.0 or 1.0) -- which equals
                # the unique value present, so y_k.mean() is correct.
                models.append(None)
                constants.append(float(y_k.mean()) if y_k.size > 0 else 0.0)
                continue
            model = LogisticRegression(
                C=self._c,
                class_weight=sk_class_weight,
                max_iter=self._max_iter,
                random_state=(self._seed + k) & _RANDOM_STATE_MASK,
                solver=_SOLVER,
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
                "predict_proba called before fit on SklearnLogisticRegressionHead"
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
