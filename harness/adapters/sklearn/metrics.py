"""``BootstrapMetrics`` -- production sklearn-backed :class:`MetricsPort`.

Behavior
--------

For each class ``k``:

* **AUROC** = :func:`sklearn.metrics.roc_auc_score` on the ``k``-th column.
  When the column collapses to a single class (no positives or no negatives)
  the score is undefined; we return ``0.5`` as a well-defined fallback.
* **AUPRC** = :func:`sklearn.metrics.average_precision_score` with the same
  fallback.
* **Precision / Recall / F1** are computed from binary predictions
  ``probs >= threshold`` versus ground-truth labels.

Macro aggregates are arithmetic means of the per-class point estimates.

Bootstrap confidence intervals
------------------------------

For ``B = bootstrap.n_resamples`` iterations we sample ``N`` row indices with
replacement from a single :class:`numpy.random.Generator` seeded by
``bootstrap.seed`` and recompute every metric on the resampled rows. The
``confidence``-level CI is the (alpha/2, 1 - alpha/2) percentile pair of the
bootstrap distribution, where ``alpha = 1 - bootstrap.confidence``. The point
estimate is the metric on the full sample.

A single :class:`~numpy.random.Generator` drives every resample, so the
output is byte-identical for the same ``(inputs, seed)`` pair.

All shape-mismatch errors funnel through
:class:`~harness.domain.errors.ContractViolation`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from harness.domain.errors import ContractViolation
from harness.domain.types import (
    BootstrapConfig,
    MetricInterval,
    MetricReport,
    PerClassMetric,
    Probabilities,
    ThresholdSet,
)

_FALLBACK_AUC: float = 0.5


def _safe_auroc(y_true: NDArray[np.int64], y_score: NDArray[np.float64]) -> float:
    if y_true.size == 0:
        return _FALLBACK_AUC
    pos = int(y_true.sum())
    if pos == 0 or pos == y_true.size:
        return _FALLBACK_AUC
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: NDArray[np.int64], y_score: NDArray[np.float64]) -> float:
    if y_true.size == 0:
        return _FALLBACK_AUC
    pos = int(y_true.sum())
    if pos == 0 or pos == y_true.size:
        return _FALLBACK_AUC
    return float(average_precision_score(y_true, y_score))


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp) / float(denom)


def _per_class_f1_vec(
    preds: NDArray[np.int64], labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    n_labels = preds.shape[1]
    out = np.zeros(n_labels, dtype=np.float64)
    for j in range(n_labels):
        p = preds[:, j]
        y = labels[:, j]
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        out[j] = _f1_from_counts(tp, fp, fn)
    return out


def _per_class_aucs(
    labels: NDArray[np.int64], probs: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_labels = labels.shape[1]
    auroc = np.zeros(n_labels, dtype=np.float64)
    auprc = np.zeros(n_labels, dtype=np.float64)
    for j in range(n_labels):
        auroc[j] = _safe_auroc(labels[:, j], probs[:, j])
        auprc[j] = _safe_auprc(labels[:, j], probs[:, j])
    return auroc, auprc


def _interval(
    point: float,
    samples: NDArray[np.float64],
    confidence: float,
) -> MetricInterval:
    """Return the raw bootstrap-quantile interval for ``point``.

    The bounds ``lo`` / ``hi`` are the unmodified ``alpha/2`` and
    ``1 - alpha/2`` quantiles of ``samples``. We do *not* widen the
    interval to bracket ``point``: doing so would silently lie about the
    bootstrap distribution. With pathological inputs (tiny ``n_resamples``
    or fully-collapsed bootstrap samples) the point estimate may legitimately
    sit outside ``[lo, hi]`` -- the :class:`MetricInterval` invariant
    accepts this case.
    """
    alpha = 1.0 - confidence
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return MetricInterval(point=point, lower=lo, upper=hi)


class BootstrapMetrics:
    """sklearn-backed multi-label metrics with bootstrap CIs."""

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

        labels_arr: NDArray[np.int64] = np.asarray(
            [[int(v) for v in row] for row in labels], dtype=np.int64
        )
        probs_arr: NDArray[np.float64] = probabilities.values.astype(np.float64)
        thr: NDArray[np.float64] = np.asarray(
            thresholds.thresholds, dtype=np.float64
        )
        preds_arr: NDArray[np.int64] = (probs_arr >= thr[None, :]).astype(np.int64)

        # Point estimates on the full sample.
        per_f1 = _per_class_f1_vec(preds_arr, labels_arr)
        per_auroc, per_auprc = _per_class_aucs(labels_arr, probs_arr)
        macro_f1 = float(per_f1.mean()) if per_f1.size else 0.0
        macro_auroc = float(per_auroc.mean()) if per_auroc.size else 0.0
        macro_auprc = float(per_auprc.mean()) if per_auprc.size else 0.0

        # Bootstrap.
        rng = np.random.default_rng(bootstrap.seed)
        n_boot = bootstrap.n_resamples
        boot_f1 = np.zeros((n_boot, n_labels), dtype=np.float64)
        boot_auroc = np.zeros((n_boot, n_labels), dtype=np.float64)
        boot_auprc = np.zeros((n_boot, n_labels), dtype=np.float64)

        for b in range(n_boot):
            idx = rng.integers(0, n_samples, size=n_samples)
            l_b = labels_arr[idx]
            p_b = probs_arr[idx]
            pred_b = preds_arr[idx]
            boot_f1[b] = _per_class_f1_vec(pred_b, l_b)
            auroc_b, auprc_b = _per_class_aucs(l_b, p_b)
            boot_auroc[b] = auroc_b
            boot_auprc[b] = auprc_b

        boot_macro_f1 = boot_f1.mean(axis=1)
        boot_macro_auroc = boot_auroc.mean(axis=1)
        boot_macro_auprc = boot_auprc.mean(axis=1)

        confidence = bootstrap.confidence

        per_class: list[PerClassMetric] = []
        for j, label in enumerate(probabilities.label_names):
            support = int(labels_arr[:, j].sum())
            per_class.append(
                PerClassMetric(
                    label=label,
                    f1=_interval(float(per_f1[j]), boot_f1[:, j], confidence),
                    auroc=_interval(
                        float(per_auroc[j]), boot_auroc[:, j], confidence
                    ),
                    auprc=_interval(
                        float(per_auprc[j]), boot_auprc[:, j], confidence
                    ),
                    support=support,
                )
            )

        return MetricReport(
            macro_f1=_interval(macro_f1, boot_macro_f1, confidence),
            macro_auroc=_interval(macro_auroc, boot_macro_auroc, confidence),
            macro_auprc=_interval(macro_auprc, boot_macro_auprc, confidence),
            per_class=tuple(per_class),
            n_bootstrap=bootstrap.n_resamples,
            seed=bootstrap.seed,
        )
