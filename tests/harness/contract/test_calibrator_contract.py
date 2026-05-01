"""Contract tests for :class:`harness.ports.calibrator.CalibratorPort`.

Per-class calibration: ``transform`` is applied independently to each column
of the probability matrix. ``is_fitted`` flips False -> True after ``fit``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.fakes.calibrator import IdentityFakeCalibrator
from harness.adapters.sklearn.calibrator import (
    PerClassIsotonicCalibrator,
    PerClassPlattCalibrator,
)
from harness.domain.errors import ContractViolation
from harness.ports.calibrator import CalibratorPort


def _make_probs_labels(
    n: int, k: int, seed: int = 0
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    rng = np.random.default_rng(seed)
    probs = rng.random(size=(n, k)).astype(np.float32)
    labels = (rng.random(size=(n, k)) > 0.5).astype(np.int8)
    return probs, labels


class CalibratorPortContract:
    """Abstract contract; subclasses provide an ``adapter`` fixture."""

    n_labels: int = 3

    @pytest.fixture
    def adapter(self) -> CalibratorPort:
        raise NotImplementedError

    def test_is_fitted_false_before_fit(self, adapter: CalibratorPort) -> None:
        assert adapter.is_fitted is False

    def test_is_fitted_true_after_fit(self, adapter: CalibratorPort) -> None:
        probs, labels = _make_probs_labels(6, self.n_labels, seed=1)
        adapter.fit(probs, labels)
        assert adapter.is_fitted is True

    def test_transform_before_fit_raises_contract_violation(
        self, adapter: CalibratorPort
    ) -> None:
        probs, _ = _make_probs_labels(5, self.n_labels, seed=2)
        with pytest.raises(ContractViolation):
            adapter.transform(probs)

    def test_transform_output_shape_matches_input(
        self, adapter: CalibratorPort
    ) -> None:
        probs, labels = _make_probs_labels(7, self.n_labels, seed=3)
        adapter.fit(probs, labels)
        out = adapter.transform(probs)
        assert out.shape == probs.shape

    def test_transform_output_in_unit_interval(
        self, adapter: CalibratorPort
    ) -> None:
        probs, labels = _make_probs_labels(7, self.n_labels, seed=4)
        adapter.fit(probs, labels)
        out = adapter.transform(probs)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_transform_is_deterministic(self, adapter: CalibratorPort) -> None:
        probs, labels = _make_probs_labels(5, self.n_labels, seed=5)
        adapter.fit(probs, labels)
        a = adapter.transform(probs)
        b = adapter.transform(probs)
        np.testing.assert_array_equal(a, b)


class TestIdentityFakeCalibratorContract(CalibratorPortContract):
    n_labels = 3

    @pytest.fixture
    def adapter(self) -> CalibratorPort:
        return IdentityFakeCalibrator()

    def test_identity_preserves_values_exactly(self) -> None:
        cal = IdentityFakeCalibrator()
        probs = np.array(
            [[0.1, 0.5, 0.9], [0.0, 1.0, 0.25]],
            dtype=np.float32,
        )
        labels = np.zeros((2, 3), dtype=np.int8)
        cal.fit(probs, labels)
        out = cal.transform(probs)
        np.testing.assert_array_equal(out, probs)

    def test_per_class_independence(self) -> None:
        """Changing column k must not affect column j != k under identity."""
        cal = IdentityFakeCalibrator()
        probs_a = np.array(
            [[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]],
            dtype=np.float32,
        )
        probs_b = probs_a.copy()
        probs_b[:, 0] = np.array([0.99, 0.01], dtype=np.float32)
        labels = np.zeros((2, 3), dtype=np.int8)
        cal.fit(probs_a, labels)
        out_a = cal.transform(probs_a)
        out_b = cal.transform(probs_b)
        # Columns 1 and 2 must match between the two transforms.
        np.testing.assert_array_equal(out_a[:, 1:], out_b[:, 1:])


class TestPerClassPlattCalibratorContract(CalibratorPortContract):
    n_labels = 3

    @pytest.fixture
    def adapter(self) -> CalibratorPort:
        return PerClassPlattCalibrator()


class TestPerClassIsotonicCalibratorContract(CalibratorPortContract):
    n_labels = 3

    @pytest.fixture
    def adapter(self) -> CalibratorPort:
        return PerClassIsotonicCalibrator()
