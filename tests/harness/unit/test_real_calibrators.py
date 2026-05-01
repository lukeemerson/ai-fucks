"""Unit tests for real per-class calibrator adapters.

Covers ``PerClassPlattCalibrator`` (sigmoid via sklearn LogisticRegression) and
``PerClassIsotonicCalibrator`` (sklearn IsotonicRegression). These adapters
implement :class:`harness.ports.calibrator.CalibratorPort` and operate on each
column of the probability matrix independently.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.sklearn.calibrator import (
    PerClassIsotonicCalibrator,
    PerClassPlattCalibrator,
)
from harness.domain.errors import ContractViolation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brier(probs: NDArray[np.float32], labels: NDArray[np.int8]) -> float:
    """Mean squared error between probabilities and labels."""
    diff = probs.astype(np.float64) - labels.astype(np.float64)
    return float(np.mean(diff * diff))


def _brier_per_class(
    probs: NDArray[np.float32], labels: NDArray[np.int8]
) -> NDArray[np.float64]:
    diff = probs.astype(np.float64) - labels.astype(np.float64)
    result: NDArray[np.float64] = np.mean(diff * diff, axis=0)
    return result


def _miscalibrated_pair(
    n: int, seed: int = 0
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    """Generate (probs, labels) where probs are systematically over-confident.

    True P(y=1 | x) = x for x ~ Uniform[0, 1]. We emit raw_prob = sqrt(x), which
    is uniformly larger than the truth on (0, 1) -> classic over-confidence.
    """
    rng = np.random.default_rng(seed)
    x = rng.random(size=(n, 1)).astype(np.float64)
    y = (rng.random(size=(n, 1)) < x).astype(np.int8)
    raw = np.sqrt(x).astype(np.float32)
    return raw, y


# ---------------------------------------------------------------------------
# Platt: monotonic preservation
# ---------------------------------------------------------------------------


class TestPlattMonotonicity:
    def test_sorted_input_maps_to_monotone_non_decreasing_output(self) -> None:
        rng = np.random.default_rng(123)
        n = 200
        x = rng.random(size=(n, 1)).astype(np.float64)
        y = (rng.random(size=(n, 1)) < x).astype(np.int8)
        probs = x.astype(np.float32)
        cal = PerClassPlattCalibrator()
        cal.fit(probs, y)

        sorted_probs = np.sort(probs, axis=0).astype(np.float32)
        out = cal.transform(sorted_probs)
        diffs = np.diff(out[:, 0])
        # Allow tiny float jitter; sigmoid composition is monotonic.
        assert float(diffs.min()) >= -1e-6


class TestIsotonicMonotonicity:
    def test_sorted_input_maps_to_monotone_non_decreasing_output(self) -> None:
        rng = np.random.default_rng(7)
        n = 200
        x = rng.random(size=(n, 1)).astype(np.float64)
        y = (rng.random(size=(n, 1)) < x).astype(np.int8)
        probs = x.astype(np.float32)
        cal = PerClassIsotonicCalibrator()
        cal.fit(probs, y)

        sorted_probs = np.sort(probs, axis=0).astype(np.float32)
        out = cal.transform(sorted_probs)
        diffs = np.diff(out[:, 0])
        assert float(diffs.min()) >= -1e-6


# ---------------------------------------------------------------------------
# Brier-score improvement on miscalibrated input
# ---------------------------------------------------------------------------


class TestPlattImprovesBrier:
    def test_brier_drops_after_calibration_on_overconfident_probs(self) -> None:
        train_probs, train_labels = _miscalibrated_pair(n=2000, seed=11)
        test_probs, test_labels = _miscalibrated_pair(n=2000, seed=12)

        cal = PerClassPlattCalibrator()
        cal.fit(train_probs, train_labels)
        calibrated = cal.transform(test_probs)

        raw_brier = _brier(test_probs, test_labels)
        cal_brier = _brier(calibrated, test_labels)
        assert cal_brier < raw_brier


class TestIsotonicImprovesBrier:
    def test_brier_drops_after_calibration_on_overconfident_probs(self) -> None:
        train_probs, train_labels = _miscalibrated_pair(n=2000, seed=21)
        test_probs, test_labels = _miscalibrated_pair(n=2000, seed=22)

        cal = PerClassIsotonicCalibrator()
        cal.fit(train_probs, train_labels)
        calibrated = cal.transform(test_probs)

        raw_brier = _brier(test_probs, test_labels)
        cal_brier = _brier(calibrated, test_labels)
        assert cal_brier < raw_brier


# ---------------------------------------------------------------------------
# Per-class independence: well-calibrated col untouched, miscalibrated col fixed
# ---------------------------------------------------------------------------


def _two_column_dataset(
    n: int, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    """Column 0 well-calibrated; column 1 over-confident (sqrt of truth)."""
    rng = np.random.default_rng(seed)
    x0 = rng.random(size=n).astype(np.float64)
    y0 = (rng.random(size=n) < x0).astype(np.int8)
    x1 = rng.random(size=n).astype(np.float64)
    y1 = (rng.random(size=n) < x1).astype(np.int8)
    raw = np.stack([x0, np.sqrt(x1)], axis=1).astype(np.float32)
    labels = np.stack([y0, y1], axis=1).astype(np.int8)
    return raw, labels


class TestPlattPerClassIndependence:
    def test_well_calibrated_column_unchanged_miscalibrated_column_fixed(
        self,
    ) -> None:
        train_probs, train_labels = _two_column_dataset(n=4000, seed=31)
        test_probs, test_labels = _two_column_dataset(n=4000, seed=32)

        cal = PerClassPlattCalibrator()
        cal.fit(train_probs, train_labels)
        calibrated = cal.transform(test_probs)

        raw_b = _brier_per_class(test_probs, test_labels)
        cal_b = _brier_per_class(calibrated, test_labels)

        # Column 0 was already well-calibrated: Brier change small.
        assert abs(float(cal_b[0]) - float(raw_b[0])) < 0.02
        # Column 1 was systematically over-confident: Brier improves.
        assert float(cal_b[1]) < float(raw_b[1]) - 0.005


class TestIsotonicPerClassIndependence:
    def test_well_calibrated_column_unchanged_miscalibrated_column_fixed(
        self,
    ) -> None:
        train_probs, train_labels = _two_column_dataset(n=4000, seed=41)
        test_probs, test_labels = _two_column_dataset(n=4000, seed=42)

        cal = PerClassIsotonicCalibrator()
        cal.fit(train_probs, train_labels)
        calibrated = cal.transform(test_probs)

        raw_b = _brier_per_class(test_probs, test_labels)
        cal_b = _brier_per_class(calibrated, test_labels)

        assert abs(float(cal_b[0]) - float(raw_b[0])) < 0.02
        assert float(cal_b[1]) < float(raw_b[1]) - 0.005


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestPlattDeterminism:
    def test_same_inputs_byte_identical_output(self) -> None:
        rng = np.random.default_rng(99)
        probs = rng.random(size=(50, 4)).astype(np.float32)
        labels = (rng.random(size=(50, 4)) > 0.5).astype(np.int8)

        a = PerClassPlattCalibrator()
        a.fit(probs, labels)
        out_a = a.transform(probs)

        b = PerClassPlattCalibrator()
        b.fit(probs, labels)
        out_b = b.transform(probs)

        np.testing.assert_array_equal(out_a, out_b)


class TestIsotonicDeterminism:
    def test_same_inputs_byte_identical_output(self) -> None:
        rng = np.random.default_rng(98)
        probs = rng.random(size=(50, 4)).astype(np.float32)
        labels = (rng.random(size=(50, 4)) > 0.5).astype(np.int8)

        a = PerClassIsotonicCalibrator()
        a.fit(probs, labels)
        out_a = a.transform(probs)

        b = PerClassIsotonicCalibrator()
        b.fit(probs, labels)
        out_b = b.transform(probs)

        np.testing.assert_array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# transform-before-fit raises ContractViolation
# ---------------------------------------------------------------------------


class TestTransformBeforeFitRaises:
    def test_platt_raises(self) -> None:
        cal = PerClassPlattCalibrator()
        probs = np.zeros((3, 2), dtype=np.float32)
        with pytest.raises(ContractViolation):
            cal.transform(probs)

    def test_isotonic_raises(self) -> None:
        cal = PerClassIsotonicCalibrator()
        probs = np.zeros((3, 2), dtype=np.float32)
        with pytest.raises(ContractViolation):
            cal.transform(probs)


# ---------------------------------------------------------------------------
# Single-class-positive / single-class-negative edge cases
# ---------------------------------------------------------------------------


class TestSingleClassEdgeCase:
    def test_platt_all_positives_does_not_crash_outputs_in_unit(self) -> None:
        rng = np.random.default_rng(5)
        probs = rng.random(size=(20, 2)).astype(np.float32)
        labels = np.ones((20, 2), dtype=np.int8)
        cal = PerClassPlattCalibrator()
        cal.fit(probs, labels)
        out = cal.transform(probs)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_platt_all_negatives_does_not_crash_outputs_in_unit(self) -> None:
        rng = np.random.default_rng(6)
        probs = rng.random(size=(20, 2)).astype(np.float32)
        labels = np.zeros((20, 2), dtype=np.int8)
        cal = PerClassPlattCalibrator()
        cal.fit(probs, labels)
        out = cal.transform(probs)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_isotonic_all_positives_does_not_crash_outputs_in_unit(self) -> None:
        rng = np.random.default_rng(8)
        probs = rng.random(size=(20, 2)).astype(np.float32)
        labels = np.ones((20, 2), dtype=np.int8)
        cal = PerClassIsotonicCalibrator()
        cal.fit(probs, labels)
        out = cal.transform(probs)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_isotonic_all_negatives_does_not_crash_outputs_in_unit(self) -> None:
        rng = np.random.default_rng(9)
        probs = rng.random(size=(20, 2)).astype(np.float32)
        labels = np.zeros((20, 2), dtype=np.int8)
        cal = PerClassIsotonicCalibrator()
        cal.fit(probs, labels)
        out = cal.transform(probs)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0


# ---------------------------------------------------------------------------
# Shape mismatch raises ContractViolation
# ---------------------------------------------------------------------------


class TestShapeMismatchRaises:
    def test_platt_shape_mismatch(self) -> None:
        probs = np.zeros((4, 3), dtype=np.float32)
        labels = np.zeros((4, 2), dtype=np.int8)
        cal = PerClassPlattCalibrator()
        with pytest.raises(ContractViolation):
            cal.fit(probs, labels)

    def test_isotonic_shape_mismatch(self) -> None:
        probs = np.zeros((4, 3), dtype=np.float32)
        labels = np.zeros((4, 2), dtype=np.int8)
        cal = PerClassIsotonicCalibrator()
        with pytest.raises(ContractViolation):
            cal.fit(probs, labels)

    def test_platt_non_2d_probs(self) -> None:
        probs = np.zeros((4,), dtype=np.float32)
        labels = np.zeros((4, 2), dtype=np.int8)
        cal = PerClassPlattCalibrator()
        with pytest.raises(ContractViolation):
            cal.fit(probs, labels)

    def test_isotonic_non_2d_probs(self) -> None:
        probs = np.zeros((4,), dtype=np.float32)
        labels = np.zeros((4, 2), dtype=np.int8)
        cal = PerClassIsotonicCalibrator()
        with pytest.raises(ContractViolation):
            cal.fit(probs, labels)
