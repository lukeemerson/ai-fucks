"""Algorithmic correctness tests for ``PrSweepShrinkageThreshold``.

These tests target the OOF PR-sweep + shrinkage-to-pooled-global algorithm
documented in ARCHITECTURE.md section 4.6 and the task brief. Each test pins
down one observable behavior of the adapter so that regressions are caught
where the algorithm matters: argmax correctness, shrinkage formula, clamp
boundaries, plateau median, per-class independence, determinism, the
``apply`` element-wise rule, and empty-class robustness.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.adapters.sklearn.threshold import PrSweepShrinkageThreshold
from harness.domain.types import (
    Predictions,
    Probabilities,
    ThresholdConfig,
    ThresholdSet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probs_from_array(values: np.ndarray, label_names: tuple[str, ...]) -> Probabilities:
    n_rows = values.shape[0]
    return Probabilities(
        sample_ids=tuple(f"s{i}" for i in range(n_rows)),
        label_names=label_names,
        values=values.astype(np.float32),
    )


def _wide_config(shrinkage: float = 1.0) -> ThresholdConfig:
    return ThresholdConfig(
        method="pr_sweep",
        shrinkage=shrinkage,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )


# ---------------------------------------------------------------------------
# 1. Synthetic-data argmax correctness
# ---------------------------------------------------------------------------


def test_argmax_picks_threshold_that_yields_perfect_f1() -> None:
    """Probabilities [0.1, 0.2, 0.45, 0.55, 0.85, 0.95], labels [0,0,0,1,1,1].

    Any threshold in (0.45, 0.55] gives perfect F1=1.0; the unique-prob
    sweep candidates include 0.55 itself, which is the smallest threshold
    that still yields F1=1.0. With plateau median, the returned threshold
    must lie in the F1=1.0 plateau.
    """
    probs_arr = np.array([[0.1], [0.2], [0.45], [0.55], [0.85], [0.95]])
    labels = [[0], [0], [0], [1], [1], [1]]
    probs = _probs_from_array(probs_arr, ("c0",))

    adapter = PrSweepShrinkageThreshold()
    ts = adapter.fit(probs, labels, config=_wide_config(shrinkage=1.0))

    t = ts.threshold_for("c0")
    # Anything in [0.46, 0.55] (inclusive of 0.55) gives perfect F1 here:
    # predictions = (probs >= t). For t in this range the last 3 are 1.
    assert 0.45 < t <= 0.55, f"expected threshold in (0.45, 0.55], got {t}"


# ---------------------------------------------------------------------------
# 2. Shrinkage formula
# ---------------------------------------------------------------------------


def test_shrinkage_formula_blends_local_and_global() -> None:
    """With shrinkage=0.5, t_shrunk = 0.5 * t_local + 0.5 * t_global.

    Strategy: build a 2-class problem where (a) at least one class has a
    local F1-optimal threshold that is non-trivially different from the
    pooled global threshold, and (b) the formula is verified on every
    class. The shrinkage formula is the load-bearing assertion; the
    differing-thresholds sanity check below guards against the test
    becoming a tautology when t_local == t_global for every class.
    """
    # Class 0: 6 negatives at low p, 6 positives at high p (separator ~0.85).
    c0_probs = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98])
    c0_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    # Class 1: positives sit at low probs (a sensible separator near 0.15
    # for the *isolated* class). Pooled probabilities span both classes.
    c1_probs = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    c1_labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    values = np.stack([c0_probs, c1_probs], axis=1)
    probs = _probs_from_array(values, ("c0", "c1"))
    labels = [[int(c0_labels[i]), int(c1_labels[i])] for i in range(len(c0_probs))]

    adapter = PrSweepShrinkageThreshold()

    # Run once with shrinkage=1.0 to recover t_local for each class.
    local = adapter.fit(probs, labels, config=_wide_config(shrinkage=1.0))
    t_local_0 = local.threshold_for("c0")
    t_local_1 = local.threshold_for("c1")

    # Run once with shrinkage=0.0 to recover t_global (both classes share it).
    pooled = adapter.fit(probs, labels, config=_wide_config(shrinkage=0.0))
    t_global = pooled.threshold_for("c0")
    assert pooled.threshold_for("c1") == pytest.approx(t_global)

    # Now run with shrinkage=0.5 and check the formula.
    blended = adapter.fit(probs, labels, config=_wide_config(shrinkage=0.5))
    t_blended_0 = blended.threshold_for("c0")
    t_blended_1 = blended.threshold_for("c1")

    assert t_blended_0 == pytest.approx(0.5 * t_local_0 + 0.5 * t_global, abs=1e-6)
    assert t_blended_1 == pytest.approx(0.5 * t_local_1 + 0.5 * t_global, abs=1e-6)
    # Sanity: at least one class threshold must differ non-trivially from
    # the pooled global; otherwise the formula assertions above are
    # vacuously satisfied (because t_local == t_global for every class).
    assert (
        abs(t_local_0 - t_global) > 0.05 or abs(t_local_1 - t_global) > 0.05
    ), "test data does not actually exercise shrinkage blending"


# ---------------------------------------------------------------------------
# 3. Clamp boundaries
# ---------------------------------------------------------------------------


def test_clamp_caps_threshold_above_upper_bound() -> None:
    """A class whose argmax F1 sits at 0.95 must be clamped to clamp_hi=0.6."""
    # Construct a class where positives only exist at very high probability so
    # the F1-optimal threshold is high.
    probs_arr = np.array(
        [[0.05], [0.10], [0.15], [0.20], [0.30], [0.40], [0.50], [0.60], [0.70], [0.95]]
    )
    labels = [[0]] * 9 + [[1]]
    probs = _probs_from_array(probs_arr, ("c0",))

    cfg = ThresholdConfig(
        method="pr_sweep",
        shrinkage=1.0,
        clamp_lo=0.1,
        clamp_hi=0.6,
    )
    adapter = PrSweepShrinkageThreshold()
    ts = adapter.fit(probs, labels, config=cfg)
    assert ts.threshold_for("c0") == pytest.approx(0.6, abs=1e-6)


def test_clamp_floors_threshold_below_lower_bound() -> None:
    """If the F1-argmax sits below clamp_lo, threshold is floored to clamp_lo."""
    # All positives at very low probability, all negatives at high probability.
    probs_arr = np.array(
        [[0.01], [0.02], [0.03], [0.50], [0.60], [0.70], [0.80], [0.90]]
    )
    labels = [[1], [1], [1], [0], [0], [0], [0], [0]]
    probs = _probs_from_array(probs_arr, ("c0",))

    cfg = ThresholdConfig(
        method="pr_sweep",
        shrinkage=1.0,
        clamp_lo=0.4,
        clamp_hi=0.9,
    )
    adapter = PrSweepShrinkageThreshold()
    ts = adapter.fit(probs, labels, config=cfg)
    assert ts.threshold_for("c0") == pytest.approx(0.4, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Plateau median
# ---------------------------------------------------------------------------


def test_plateau_returns_median_of_tied_thresholds() -> None:
    """When several thresholds tie at max F1, return the median of the tie."""
    # Probabilities are perfectly separated: negatives in [0.10..0.30], one
    # positive at 0.70. Every threshold strictly between 0.30 and 0.70 (and
    # equal to 0.70) produces the same predictions and thus the same F1=1.0.
    # Candidate thresholds in that plateau include linspace points at
    # 0.31..0.69 and the unique-prob 0.70. The median should be ~0.50.
    probs_arr = np.array([[0.10], [0.20], [0.30], [0.70]])
    labels = [[0], [0], [0], [1]]
    probs = _probs_from_array(probs_arr, ("c0",))

    adapter = PrSweepShrinkageThreshold()
    ts = adapter.fit(probs, labels, config=_wide_config(shrinkage=1.0))
    t = ts.threshold_for("c0")
    # Plateau spans roughly (0.30, 0.70]; median of candidates in that range
    # should be near 0.50, definitely not at the min (~0.31) or the max (0.70).
    assert 0.45 <= t <= 0.55, f"expected median ~0.5 within plateau, got {t}"


# ---------------------------------------------------------------------------
# 5. Per-class independence
# ---------------------------------------------------------------------------


def test_changing_class0_probs_does_not_change_class1_threshold() -> None:
    """Per-class fitting: class 0's data must not leak into class 1's threshold.

    Note: the global pooled anchor depends on all classes, so to pin per-class
    independence we run with shrinkage=1.0 (no global blending).
    """
    # Class 1 stays fixed across both runs.
    c1_probs = np.array([0.05, 0.10, 0.15, 0.80, 0.85, 0.90])
    c1_labels = np.array([0, 0, 0, 1, 1, 1])

    # First run: class 0 has one separation pattern.
    c0_probs_a = np.array([0.10, 0.20, 0.30, 0.70, 0.80, 0.90])
    c0_labels_a = np.array([0, 0, 0, 1, 1, 1])

    # Second run: class 0 has totally different probabilities, but class 1
    # is identical.
    c0_probs_b = np.array([0.40, 0.45, 0.48, 0.52, 0.55, 0.60])
    c0_labels_b = np.array([1, 0, 1, 0, 1, 0])  # noisy

    values_a = np.stack([c0_probs_a, c1_probs], axis=1)
    values_b = np.stack([c0_probs_b, c1_probs], axis=1)
    probs_a = _probs_from_array(values_a, ("c0", "c1"))
    probs_b = _probs_from_array(values_b, ("c0", "c1"))
    labels_a = [[int(c0_labels_a[i]), int(c1_labels[i])] for i in range(6)]
    labels_b = [[int(c0_labels_b[i]), int(c1_labels[i])] for i in range(6)]

    adapter = PrSweepShrinkageThreshold()
    cfg = _wide_config(shrinkage=1.0)
    ts_a = adapter.fit(probs_a, labels_a, config=cfg)
    ts_b = adapter.fit(probs_b, labels_b, config=cfg)

    assert ts_a.threshold_for("c1") == pytest.approx(ts_b.threshold_for("c1"))


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


def test_fit_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    values = rng.random((20, 4)).astype(np.float32)
    probs = _probs_from_array(values, ("a", "b", "c", "d"))
    labels = [[(i + j) % 2 for j in range(4)] for i in range(20)]

    cfg = ThresholdConfig(
        method="pr_sweep", shrinkage=0.3, clamp_lo=0.05, clamp_hi=0.95
    )
    adapter = PrSweepShrinkageThreshold()
    a = adapter.fit(probs, labels, config=cfg)
    b = adapter.fit(probs, labels, config=cfg)
    assert a.thresholds == b.thresholds
    assert a.label_names == b.label_names


# ---------------------------------------------------------------------------
# 7. Apply correctness
# ---------------------------------------------------------------------------


def test_apply_produces_correct_binary_predictions() -> None:
    values = np.array(
        [[0.10, 0.90], [0.50, 0.50], [0.80, 0.20]], dtype=np.float32
    )
    probs = _probs_from_array(values, ("c0", "c1"))
    ts = ThresholdSet(
        label_names=("c0", "c1"),
        thresholds=(0.5, 0.5),
        method="manual",
        shrinkage=0.0,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )
    adapter = PrSweepShrinkageThreshold()
    preds = adapter.apply(probs, ts)
    assert isinstance(preds, Predictions)
    expected = np.array([[0, 1], [1, 1], [1, 0]], dtype=np.int8)
    assert np.array_equal(preds.values, expected)


def test_apply_uses_per_label_threshold_lookup_by_name() -> None:
    """``apply`` must look up thresholds by label name, not by position."""
    values = np.array([[0.40, 0.40], [0.60, 0.60]], dtype=np.float32)
    probs = _probs_from_array(values, ("c0", "c1"))
    # Build threshold set with the SAME labels so the contract stays valid;
    # different per-class thresholds prove the per-class lookup is exercised.
    ts = ThresholdSet(
        label_names=("c0", "c1"),
        thresholds=(0.5, 0.7),
        method="manual",
        shrinkage=0.0,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )
    adapter = PrSweepShrinkageThreshold()
    preds = adapter.apply(probs, ts)
    # c0: [0.40>=0.5 -> 0, 0.60>=0.5 -> 1]
    # c1: [0.40>=0.7 -> 0, 0.60>=0.7 -> 0]
    assert preds.values[0, 0] == 0
    assert preds.values[1, 0] == 1
    assert preds.values[0, 1] == 0
    assert preds.values[1, 1] == 0


# ---------------------------------------------------------------------------
# 8. Empty-class robustness
# ---------------------------------------------------------------------------


def test_all_negative_class_falls_back_to_global_pooled_threshold() -> None:
    """A class with zero positives has no F1 signal; fall back to pooled global.

    Wave 6a updated the empty-class fallback from ``clamp_hi`` to the pooled
    global F1-optimal threshold (computed across every class's probabilities
    and labels concatenated). With ``shrinkage=1.0`` the blended formula
    reduces to ``t_local`` -- which itself was substituted with the pooled
    global -- so the result is the pooled global, *not* the upper clamp.
    """
    values = np.array(
        [[0.10, 0.10], [0.30, 0.30], [0.50, 0.50], [0.70, 0.70], [0.90, 0.90]],
        dtype=np.float32,
    )
    probs = _probs_from_array(values, ("c0", "c1"))
    # c0 has positives that progress 0.1 -> 0.9; c1 has all-zero labels.
    labels = [[0, 0], [0, 0], [1, 0], [1, 0], [1, 0]]

    cfg = ThresholdConfig(
        method="pr_sweep", shrinkage=1.0, clamp_lo=0.05, clamp_hi=0.85
    )
    adapter = PrSweepShrinkageThreshold()
    ts = adapter.fit(probs, labels, config=cfg)

    # Compute the expected pooled-global threshold by running the same PR
    # sweep over the pooled probs/labels: this is exactly what the adapter
    # uses for its empty-class fallback.
    pooled_probs = np.asarray(values, dtype=np.float64).reshape(-1)
    pooled_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    expected_pooled = PrSweepShrinkageThreshold._argmax_f1_threshold(
        pooled_probs, pooled_labels
    )
    assert not np.isnan(expected_pooled), "test setup needs a defined pooled threshold"
    expected_pooled_clamped = float(
        np.clip(expected_pooled, cfg.clamp_lo, cfg.clamp_hi)
    )

    # Empty class falls back to the pooled global value.
    assert ts.threshold_for("c1") == pytest.approx(
        expected_pooled_clamped, abs=1e-6
    )
    # And that pooled value is meaningfully different from clamp_hi -- otherwise
    # this assertion would be vacuously satisfied by the old clamp_hi behavior.
    assert abs(expected_pooled_clamped - cfg.clamp_hi) > 0.05, (
        "test data does not exercise the new pooled-global fallback "
        "(pooled value happens to coincide with clamp_hi)"
    )

    # Sanity: c0 has its own signal and must not collapse to clamp_hi either.
    assert ts.threshold_for("c0") != pytest.approx(cfg.clamp_hi, abs=1e-6)


def test_all_classes_empty_falls_back_to_clamp_midpoint() -> None:
    """When no class has any positive labels the pooled column also has no
    F1 signal. The second-tier fallback is the midpoint of [clamp_lo, clamp_hi].

    This pins down the Wave 6a midpoint fallback that the previous test does
    *not* exercise (because c0 in that test has positives that yield a defined
    pooled global threshold).
    """
    values = np.array(
        [[0.10, 0.10], [0.30, 0.30], [0.50, 0.50], [0.70, 0.70], [0.90, 0.90]],
        dtype=np.float32,
    )
    probs = _probs_from_array(values, ("c0", "c1"))
    # Both classes have all-zero labels: pooled column also has no positives.
    labels = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    cfg = ThresholdConfig(
        method="pr_sweep", shrinkage=1.0, clamp_lo=0.10, clamp_hi=0.80
    )
    midpoint = 0.5 * (cfg.clamp_lo + cfg.clamp_hi)

    adapter = PrSweepShrinkageThreshold()
    ts = adapter.fit(probs, labels, config=cfg)

    # Both classes (and the pooled column) have no signal -> midpoint fallback.
    assert ts.threshold_for("c0") == pytest.approx(midpoint, abs=1e-6)
    assert ts.threshold_for("c1") == pytest.approx(midpoint, abs=1e-6)
    # And the midpoint is strictly inside the clamp range, not on the boundary.
    assert cfg.clamp_lo < ts.threshold_for("c0") < cfg.clamp_hi
