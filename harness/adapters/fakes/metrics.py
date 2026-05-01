"""``CountingFakeMetrics`` -- deterministic, dependency-free metrics adapter.

Behavior
--------

* Per-class precision / recall / F1 are computed exactly via numpy TP/FP/FN
  counts. These values matter because downstream code (model cards,
  reproducibility tests) compares them.
* Per-class AUROC and AUPRC are *not* implemented in this fake. Computing
  them correctly without sklearn is non-trivial and irrelevant to what this
  fake exists to do (let contract and integration tests run without sklearn).
  We return ``0.5`` as a constant point estimate with a zero-width interval.
  The production sklearn adapter computes the real numbers.
* Bootstrap confidence intervals are returned as zero-width ``(point, point,
  point)``. This is deterministic and trivially honours the
  ``lower <= point <= upper`` invariant.
* The macro F1/AUROC/AUPRC point estimates are the arithmetic mean of the
  per-class point estimates; intervals follow the same zero-width scheme.

All shape-mismatch errors are funnelled through
:class:`~harness.domain.errors.ContractViolation`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from harness.domain.errors import ContractViolation
from harness.domain.types import (
    BootstrapConfig,
    MetricInterval,
    MetricReport,
    PerClassMetric,
    Probabilities,
    ThresholdSet,
)

_CONST_AUC: float = 0.5  # see module docstring for rationale.


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        # No positives in either preds or labels: define F1 as 1.0 (perfect
        # all-zero classifier on an all-zero column). This keeps the macro
        # mean defined when a label is fully absent.
        return 1.0
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp) / float(denom)


def _per_class_f1(
    preds: NDArray[np.int8], labels: NDArray[np.int8]
) -> list[float]:
    n_labels = preds.shape[1]
    out: list[float] = []
    for j in range(n_labels):
        p = preds[:, j].astype(np.int64)
        y = labels[:, j].astype(np.int64)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        out.append(_f1_from_counts(tp, fp, fn))
    return out


def _zero_width(point: float) -> MetricInterval:
    return MetricInterval(point=point, lower=point, upper=point)


class CountingFakeMetrics:
    """Plain numpy TP/FP/FN F1; constant 0.5 AUROC/AUPRC; zero-width CIs."""

    def evaluate(
        self,
        probabilities: Probabilities,
        labels: Sequence[Sequence[int]],
        thresholds: ThresholdSet,
        *,
        bootstrap: BootstrapConfig,
    ) -> MetricReport:
        if thresholds.label_names != probabilities.label_names:
            raise ContractViolation(
                "thresholds.label_names "
                f"{thresholds.label_names!r} != probabilities.label_names "
                f"{probabilities.label_names!r}"
            )
        n_samples, n_labels = probabilities.values.shape
        if len(labels) != n_samples:
            raise ContractViolation(
                f"len(labels) {len(labels)} != n_samples {n_samples}"
            )
        for i, row in enumerate(labels):
            if len(row) != n_labels:
                raise ContractViolation(
                    f"labels[{i}] has length {len(row)} != n_labels {n_labels}"
                )

        labels_arr = np.asarray(
            [[int(v) for v in row] for row in labels], dtype=np.int8
        )
        thr = np.asarray(thresholds.thresholds, dtype=np.float32)
        preds = (probabilities.values >= thr[None, :]).astype(np.int8)

        per_class_f1 = _per_class_f1(preds, labels_arr)
        # support: number of positive labels per class.
        supports: list[int] = [
            int(labels_arr[:, j].sum()) for j in range(n_labels)
        ]

        per_class: list[PerClassMetric] = []
        for j, label in enumerate(probabilities.label_names):
            per_class.append(
                PerClassMetric(
                    label=label,
                    f1=_zero_width(per_class_f1[j]),
                    auroc=_zero_width(_CONST_AUC),
                    auprc=_zero_width(_CONST_AUC),
                    support=supports[j],
                )
            )

        macro_f1_point = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
        return MetricReport(
            macro_f1=_zero_width(macro_f1_point),
            macro_auroc=_zero_width(_CONST_AUC),
            macro_auprc=_zero_width(_CONST_AUC),
            per_class=tuple(per_class),
            n_bootstrap=bootstrap.n_resamples,
            seed=bootstrap.seed,
        )
