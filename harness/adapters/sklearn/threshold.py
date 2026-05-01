"""``PrSweepShrinkageThreshold`` -- production ``ThresholdPort`` adapter.

Algorithm (per ARCHITECTURE.md section 4.6 and the project brief)
-----------------------------------------------------------------

For each class ``k``:

1. Sweep candidates ``numpy.linspace(0.01, 0.99, 199)`` plus all unique
   probabilities in OOF column ``k`` (deduplicated, sorted).
2. For each candidate ``t``: predictions = ``probs[:, k] >= t``; compute
   ``f1 = 2*tp / (2*tp + fp + fn)`` (zero when the denominator is zero).
3. ``t_local_k = argmax F1(t)``. Plateau-smoothing: when several thresholds
   tie at the maximum F1, pick the **median** of the tied set (stability).
4. ``t_global`` = same sweep applied to the **pooled** OOF column (all
   probabilities and labels concatenated across classes).
5. ``t_shrunk_k = lambda * t_local_k + (1 - lambda) * t_global`` where
   ``lambda = config.shrinkage``.
6. ``t_final_k = clip(t_shrunk_k, config.clamp_lo, config.clamp_hi)``.

Empty-class robustness: a class with zero positives carries no F1 signal.
Its ``t_local`` falls back to the pooled global threshold so the shrinkage
formula reduces to ``t_global`` itself; if the pooled column also has no
positive signal the fallback is the midpoint of ``[clamp_lo, clamp_hi]``.
The result still flows through the standard shrinkage-and-clamp pipeline.

``apply(probs, thresholds)`` looks each label's threshold up by **name** and
returns ``(probs[:, k] >= threshold[k])`` element-wise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from harness.domain.errors import ContractViolation
from harness.domain.types import (
    Predictions,
    Probabilities,
    ThresholdConfig,
    ThresholdSet,
)

_LINSPACE_LO: Final[float] = 0.01
_LINSPACE_HI: Final[float] = 0.99
_LINSPACE_N: Final[int] = 199
_METHOD_NAME: Final[str] = "pr_sweep_shrinkage"

# Sentinel returned by ``_argmax_f1_threshold`` when the input column carries
# no positive signal and therefore offers no F1-meaningful threshold.
_NO_SIGNAL: Final[float] = float("nan")


@dataclass(frozen=True)
class PrSweepShrinkageThreshold:
    """OOF PR-sweep with shrinkage to the pooled global threshold."""

    @property
    def identifier(self) -> str:
        return "sklearn-pr-sweep-shrinkage"

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        calibrated_oof: Probabilities,
        labels: Sequence[Sequence[int]],
        *,
        config: ThresholdConfig,
    ) -> ThresholdSet:
        labels_arr = self._labels_to_array(labels, calibrated_oof)
        probs_arr = calibrated_oof.values.astype(np.float64, copy=False)

        midpoint = 0.5 * (config.clamp_lo + config.clamp_hi)

        # Pooled global threshold from the concatenated columns. If the pooled
        # column also has no positive signal, fall back to the clamp midpoint
        # rather than the upper clamp -- the latter biases every class toward
        # never-firing in the absence of any data signal.
        pooled_probs = probs_arr.reshape(-1)
        pooled_labels = labels_arr.reshape(-1)
        t_global_raw = self._argmax_f1_threshold(pooled_probs, pooled_labels)
        t_global = midpoint if np.isnan(t_global_raw) else t_global_raw

        n_labels = probs_arr.shape[1]
        thresholds: list[float] = []
        lam = config.shrinkage
        for k in range(n_labels):
            col_probs = probs_arr[:, k]
            col_labels = labels_arr[:, k]
            t_local_raw = self._argmax_f1_threshold(col_probs, col_labels)
            # Empty-class fallback: use the pooled global threshold so that
            # ``lam * t_local + (1 - lam) * t_global`` reduces to ``t_global``.
            t_local = t_global if np.isnan(t_local_raw) else t_local_raw
            t_shrunk = lam * t_local + (1.0 - lam) * t_global
            t_final = float(np.clip(t_shrunk, config.clamp_lo, config.clamp_hi))
            thresholds.append(t_final)

        return ThresholdSet(
            label_names=calibrated_oof.label_names,
            thresholds=tuple(thresholds),
            method=_METHOD_NAME,
            shrinkage=config.shrinkage,
            clamp_lo=config.clamp_lo,
            clamp_hi=config.clamp_hi,
        )

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

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
        # Lookup by name to honor the per-class lookup contract even when
        # label_names match positionally; this also future-proofs against
        # callers passing a re-ordered ThresholdSet.
        ordered = np.asarray(
            [thresholds.threshold_for(name) for name in probabilities.label_names],
            dtype=np.float32,
        )
        binary = (probabilities.values >= ordered[None, :]).astype(np.int8)
        return Predictions(
            sample_ids=probabilities.sample_ids,
            label_names=probabilities.label_names,
            values=binary,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _labels_to_array(
        labels: Sequence[Sequence[int]],
        calibrated_oof: Probabilities,
    ) -> NDArray[np.int64]:
        n_rows, n_cols = calibrated_oof.values.shape
        if len(labels) != n_rows:
            raise ContractViolation(
                f"len(labels) {len(labels)} != n_rows {n_rows}"
            )
        out = np.zeros((n_rows, n_cols), dtype=np.int64)
        for i, row in enumerate(labels):
            if len(row) != n_cols:
                raise ContractViolation(
                    f"labels[{i}] length {len(row)} != n_cols {n_cols}"
                )
            for j, v in enumerate(row):
                if v not in (0, 1):
                    raise ContractViolation(
                        f"labels[{i}][{j}] must be 0 or 1, got {v}"
                    )
                out[i, j] = int(v)
        return out

    @staticmethod
    def _candidate_thresholds(probs: NDArray[np.float64]) -> NDArray[np.float64]:
        linspace = np.linspace(_LINSPACE_LO, _LINSPACE_HI, _LINSPACE_N)
        if probs.size == 0:
            return linspace.astype(np.float64)
        unique = np.unique(probs.astype(np.float64))
        merged = np.unique(np.concatenate([linspace, unique]))
        return merged.astype(np.float64)

    @classmethod
    def _argmax_f1_threshold(
        cls,
        probs: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> float:
        """Return the F1-argmax threshold (median of plateau ties).

        Returns :data:`_NO_SIGNAL` (NaN) when the column carries no positive
        signal -- either it is empty, has no positive labels, or no candidate
        threshold yields any true-positive overlap. Callers decide which
        fallback (pooled global, clamp midpoint, etc.) to substitute.
        """
        if probs.size == 0 or int(labels.sum()) == 0:
            return _NO_SIGNAL

        candidates = cls._candidate_thresholds(probs)
        # Vectorize: predictions[i, t] = (probs[i] >= candidates[t]).
        preds = probs[:, None] >= candidates[None, :]
        labels_col = labels.astype(np.int64)[:, None]
        tp = np.sum(preds & (labels_col == 1), axis=0).astype(np.float64)
        fp = np.sum(preds & (labels_col == 0), axis=0).astype(np.float64)
        fn = np.sum((~preds) & (labels_col == 1), axis=0).astype(np.float64)
        denom = 2.0 * tp + fp + fn
        f1 = np.where(denom > 0.0, 2.0 * tp / np.where(denom > 0.0, denom, 1.0), 0.0)

        max_f1 = float(np.max(f1))
        if max_f1 <= 0.0:
            # No threshold yields any positive prediction overlap with truth;
            # the F1 surface is flat at 0. No signal -- defer to caller.
            return _NO_SIGNAL

        # Plateau median: among all candidates whose F1 equals the maximum,
        # return the median (stable tie-breaking; matters when the F1 surface
        # has long flat regions, which is common with discrete labels).
        tied = candidates[f1 >= max_f1 - 1e-12]
        return float(np.median(tied))
