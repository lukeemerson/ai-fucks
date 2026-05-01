"""Unit tests for :class:`harness.adapters.sklearn.metrics.BootstrapMetrics`.

These tests pin down the *numerical* behavior of the production sklearn-backed
metrics adapter independently of the port contract suite. They exercise:

* Macro-F1 = arithmetic mean of per-class F1.
* AUROC point estimates on perfectly separable and on near-random inputs.
* Bootstrap CI invariants (bracketing, determinism, seed sensitivity).
* Robustness to single-class label columns.
* Error funneling through :class:`ContractViolation`.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.sklearn.metrics import BootstrapMetrics
from harness.domain.errors import ContractViolation
from harness.domain.types import (
    BootstrapConfig,
    Probabilities,
    ThresholdSet,
)


def _make_probs(
    values: NDArray[np.float32],
    label_names: tuple[str, ...] | None = None,
) -> Probabilities:
    n_rows, n_cols = values.shape
    names = label_names or tuple(f"l{j}" for j in range(n_cols))
    return Probabilities(
        sample_ids=tuple(f"s{i}" for i in range(n_rows)),
        label_names=names,
        values=values,
    )


def _make_thresholds(label_names: tuple[str, ...]) -> ThresholdSet:
    return ThresholdSet(
        label_names=label_names,
        thresholds=tuple(0.5 for _ in label_names),
        method="manual",
        shrinkage=0.0,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )


def _bootstrap_cfg(seed: int = 11, n: int = 32) -> BootstrapConfig:
    return BootstrapConfig(n_resamples=n, confidence=0.95, seed=seed)


# ---------------------------------------------------------------------------
# 1. Macro-F1 = mean of per-class F1
# ---------------------------------------------------------------------------


def test_macro_f1_equals_mean_of_per_class_f1() -> None:
    # 3-class problem with hand-computable F1s.
    # Threshold = 0.5 → preds = (probs >= 0.5).
    probs_arr = np.array(
        [
            [0.9, 0.1, 0.9],  # preds 1,0,1
            [0.9, 0.9, 0.1],  # preds 1,1,0
            [0.1, 0.1, 0.9],  # preds 0,0,1
            [0.1, 0.9, 0.1],  # preds 0,1,0
        ],
        dtype=np.float32,
    )
    labels = [
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [0, 1, 0],
    ]
    probs = _make_probs(probs_arr)
    adapter = BootstrapMetrics()
    report = adapter.evaluate(
        probs,
        labels,
        _make_thresholds(probs.label_names),
        bootstrap=_bootstrap_cfg(),
    )

    expected_macro = sum(c.f1.point for c in report.per_class) / len(
        report.per_class
    )
    assert report.macro_f1.point == pytest.approx(expected_macro, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. AUROC = 1.0 on perfectly separable input
# ---------------------------------------------------------------------------


def test_auroc_perfectly_separable_is_one() -> None:
    probs_arr = np.array(
        [[0.1], [0.2], [0.8], [0.9]], dtype=np.float32
    )
    labels = [[0], [0], [1], [1]]
    probs = _make_probs(probs_arr, label_names=("only",))
    adapter = BootstrapMetrics()
    report = adapter.evaluate(
        probs,
        labels,
        _make_thresholds(probs.label_names),
        bootstrap=_bootstrap_cfg(),
    )
    assert report.per_class[0].auroc.point == pytest.approx(1.0, abs=1e-12)
    assert report.per_class[0].auprc.point == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. AUROC ~ 0.5 on random labels
# ---------------------------------------------------------------------------


def test_auroc_random_labels_near_one_half() -> None:
    rng = np.random.default_rng(123)
    n = 4000
    probs_arr = rng.random((n, 1)).astype(np.float32)
    labels_int = rng.integers(0, 2, size=n).tolist()
    labels = [[int(v)] for v in labels_int]
    probs = _make_probs(probs_arr, label_names=("only",))
    adapter = BootstrapMetrics()
    report = adapter.evaluate(
        probs,
        labels,
        _make_thresholds(probs.label_names),
        bootstrap=_bootstrap_cfg(n=4),
    )
    assert report.per_class[0].auroc.point == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# 4. Bootstrap CI brackets point estimate
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(7)
    probs_arr = rng.random((50, 4)).astype(np.float32)
    labels = [
        [int(v) for v in rng.integers(0, 2, size=4)] for _ in range(50)
    ]
    probs = _make_probs(probs_arr)
    adapter = BootstrapMetrics()
    report = adapter.evaluate(
        probs,
        labels,
        _make_thresholds(probs.label_names),
        bootstrap=_bootstrap_cfg(n=64),
    )
    for cls in report.per_class:
        for interval in (cls.f1, cls.auroc, cls.auprc):
            assert interval.lower <= interval.point <= interval.upper
    for interval in (
        report.macro_f1,
        report.macro_auroc,
        report.macro_auprc,
    ):
        assert interval.lower <= interval.point <= interval.upper


# ---------------------------------------------------------------------------
# 5. Bootstrap determinism: same seed → identical bounds
# ---------------------------------------------------------------------------


def test_bootstrap_determinism_same_seed() -> None:
    rng = np.random.default_rng(0)
    probs_arr = rng.random((30, 3)).astype(np.float32)
    labels = [
        [int(v) for v in rng.integers(0, 2, size=3)] for _ in range(30)
    ]
    probs = _make_probs(probs_arr)
    ts = _make_thresholds(probs.label_names)
    cfg = _bootstrap_cfg(seed=99, n=32)

    adapter = BootstrapMetrics()
    a = adapter.evaluate(probs, labels, ts, bootstrap=cfg)
    b = adapter.evaluate(probs, labels, ts, bootstrap=cfg)

    assert a.macro_f1.point == b.macro_f1.point
    assert a.macro_f1.lower == b.macro_f1.lower
    assert a.macro_f1.upper == b.macro_f1.upper
    assert a.macro_auroc.lower == b.macro_auroc.lower
    assert a.macro_auroc.upper == b.macro_auroc.upper
    assert a.macro_auprc.lower == b.macro_auprc.lower
    assert a.macro_auprc.upper == b.macro_auprc.upper
    for ca, cb in zip(a.per_class, b.per_class, strict=True):
        for ia, ib in (
            (ca.f1, cb.f1),
            (ca.auroc, cb.auroc),
            (ca.auprc, cb.auprc),
        ):
            assert ia.point == ib.point
            assert ia.lower == ib.lower
            assert ia.upper == ib.upper


# ---------------------------------------------------------------------------
# 6. Different seed → different bounds
# ---------------------------------------------------------------------------


def test_bootstrap_different_seed_changes_bounds() -> None:
    rng = np.random.default_rng(1)
    probs_arr = rng.random((40, 3)).astype(np.float32)
    labels = [
        [int(v) for v in rng.integers(0, 2, size=3)] for _ in range(40)
    ]
    probs = _make_probs(probs_arr)
    ts = _make_thresholds(probs.label_names)

    adapter = BootstrapMetrics()
    a = adapter.evaluate(
        probs, labels, ts, bootstrap=_bootstrap_cfg(seed=1, n=64)
    )
    b = adapter.evaluate(
        probs, labels, ts, bootstrap=_bootstrap_cfg(seed=2, n=64)
    )

    # Point estimates should be identical (data unchanged); bounds differ.
    assert a.macro_f1.point == pytest.approx(b.macro_f1.point, abs=1e-12)

    # At least one bound should differ between seeds for at least one class
    # metric. We check macro-level bounds plus per-class bounds.
    diffs: list[bool] = []
    for ia, ib in (
        (a.macro_f1, b.macro_f1),
        (a.macro_auroc, b.macro_auroc),
        (a.macro_auprc, b.macro_auprc),
    ):
        diffs.append(ia.lower != ib.lower or ia.upper != ib.upper)
    for ca, cb in zip(a.per_class, b.per_class, strict=True):
        for ia, ib in (
            (ca.f1, cb.f1),
            (ca.auroc, cb.auroc),
            (ca.auprc, cb.auprc),
        ):
            diffs.append(ia.lower != ib.lower or ia.upper != ib.upper)
    assert any(diffs), "different seeds must shift at least one CI bound"


# ---------------------------------------------------------------------------
# 7. Single-class column robustness
# ---------------------------------------------------------------------------


def test_single_class_column_does_not_crash() -> None:
    # Column 1 is all-zero labels: AUROC undefined → fallback 0.5; F1 = 0.0
    # (no positives in labels, but preds may still fire).
    probs_arr = np.array(
        [
            [0.9, 0.6],
            [0.1, 0.7],
            [0.8, 0.4],
            [0.2, 0.9],
        ],
        dtype=np.float32,
    )
    labels = [
        [1, 0],
        [0, 0],
        [1, 0],
        [0, 0],
    ]
    probs = _make_probs(probs_arr)
    adapter = BootstrapMetrics()
    report = adapter.evaluate(
        probs,
        labels,
        _make_thresholds(probs.label_names),
        bootstrap=_bootstrap_cfg(n=8),
    )
    # Column 1: all-zero labels.
    cls1 = report.per_class[1]
    assert cls1.auroc.point == pytest.approx(0.5)
    assert cls1.auprc.point == pytest.approx(0.5)
    assert cls1.f1.point == pytest.approx(0.0)
    assert cls1.support == 0


# ---------------------------------------------------------------------------
# 8. Shape mismatch raises ContractViolation
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises_contract_violation() -> None:
    probs_arr = np.zeros((4, 3), dtype=np.float32)
    probs = _make_probs(probs_arr)
    # Label rows have wrong width.
    bad_labels = [[0, 1] for _ in range(4)]
    adapter = BootstrapMetrics()
    with pytest.raises(ContractViolation):
        adapter.evaluate(
            probs,
            bad_labels,
            _make_thresholds(probs.label_names),
            bootstrap=_bootstrap_cfg(),
        )


def test_row_count_mismatch_raises_contract_violation() -> None:
    probs_arr = np.zeros((4, 3), dtype=np.float32)
    probs = _make_probs(probs_arr)
    too_few_labels = [[0, 0, 0] for _ in range(3)]
    adapter = BootstrapMetrics()
    with pytest.raises(ContractViolation):
        adapter.evaluate(
            probs,
            too_few_labels,
            _make_thresholds(probs.label_names),
            bootstrap=_bootstrap_cfg(),
        )


def test_thresholds_label_mismatch_raises_contract_violation() -> None:
    probs_arr = np.zeros((4, 3), dtype=np.float32)
    probs = _make_probs(probs_arr)
    bad_ts = ThresholdSet(
        label_names=("a", "b"),
        thresholds=(0.5, 0.5),
        method="manual",
        shrinkage=0.0,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )
    adapter = BootstrapMetrics()
    with pytest.raises(ContractViolation):
        adapter.evaluate(
            probs,
            [[0, 0, 0] for _ in range(4)],
            bad_ts,
            bootstrap=_bootstrap_cfg(),
        )
