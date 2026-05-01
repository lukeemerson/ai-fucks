"""Unit tests for :class:`SklearnGradientBoostingHead`.

Behavior assertions only: each test asserts on inputs/outputs through the
``ClassifierHeadPort`` surface (``fit`` / ``predict_proba``). No private
attribute access. Test data is intentionally tiny (16 rows, 4 features,
3 classes) so the entire file finishes in well under a second.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.sklearn.head import SklearnGradientBoostingHead
from harness.domain.errors import ContractViolation

# Fast fit budget for tests that don't need a real learning signal. The
# adapter's production default (200) is unnecessary for shape / dtype /
# determinism / error-handling assertions; bumping it down keeps the
# default suite under the < 5s wall-clock budget.
_FAST_MAX_ITER: int = 15


def _make_xy(
    n: int, d: int, k: int, *, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n, d)).astype(np.float32)
    y = (rng.random(size=(n, k)) > 0.5).astype(np.int8)
    return x, y


def _make_separable_xy(
    n: int, d: int, k: int, *, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    """Build a near-linearly-separable multi-label dataset.

    Each class ``j`` fires when feature ``x[:, j % d]`` exceeds zero. The
    classifier should be able to learn this trivially, giving us a strong
    signal for ranking/probability tests.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n, d)).astype(np.float32)
    y = np.zeros((n, k), dtype=np.int8)
    for j in range(k):
        y[:, j] = (x[:, j % d] > 0.0).astype(np.int8)
    return x, y


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_zero_n_labels_rejected(self) -> None:
        with pytest.raises(ContractViolation):
            SklearnGradientBoostingHead(n_labels=0, seed=0)

    def test_negative_n_labels_rejected(self) -> None:
        with pytest.raises(ContractViolation):
            SklearnGradientBoostingHead(n_labels=-1, seed=0)


# ---------------------------------------------------------------------------
# Fit-input validation
# ---------------------------------------------------------------------------


class TestFitInputValidation:
    def test_features_must_be_2d(self) -> None:
        head = SklearnGradientBoostingHead(n_labels=3, seed=0)
        x = np.zeros((10,), dtype=np.float32)
        y = np.zeros((10, 3), dtype=np.int8)
        with pytest.raises(ContractViolation):
            head.fit(x, y)

    def test_labels_must_be_2d(self) -> None:
        head = SklearnGradientBoostingHead(n_labels=3, seed=0)
        x = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros((10,), dtype=np.int8)
        with pytest.raises(ContractViolation):
            head.fit(x, y)

    def test_row_mismatch_rejected(self) -> None:
        head = SklearnGradientBoostingHead(n_labels=3, seed=0)
        x = np.zeros((8, 4), dtype=np.float32)
        y = np.zeros((10, 3), dtype=np.int8)
        with pytest.raises(ContractViolation):
            head.fit(x, y)

    def test_label_count_mismatch_rejected(self) -> None:
        head = SklearnGradientBoostingHead(n_labels=3, seed=0)
        x = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros((10, 4), dtype=np.int8)  # 4 != n_labels=3
        with pytest.raises(ContractViolation):
            head.fit(x, y)


# ---------------------------------------------------------------------------
# Predict-input validation
# ---------------------------------------------------------------------------


class TestPredictInputValidation:
    def test_predict_before_fit_raises(self) -> None:
        head = SklearnGradientBoostingHead(n_labels=3, seed=0)
        x = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ContractViolation):
            head.predict_proba(x)

    def test_predict_features_must_be_2d(self) -> None:
        head = SklearnGradientBoostingHead(
            n_labels=3, seed=0, max_iter=_FAST_MAX_ITER
        )
        x_train, y_train = _make_separable_xy(16, 4, 3, seed=0)
        head.fit(x_train, y_train)
        x_bad = np.zeros((5,), dtype=np.float32)
        with pytest.raises(ContractViolation):
            head.predict_proba(x_bad)


# ---------------------------------------------------------------------------
# Output-shape / dtype contracts
# ---------------------------------------------------------------------------


class TestOutputShapeAndDtype:
    def test_predict_proba_shape_matches_n_rows_by_n_labels(self) -> None:
        head = SklearnGradientBoostingHead(
            n_labels=3, seed=0, max_iter=_FAST_MAX_ITER
        )
        x, y = _make_separable_xy(16, 4, 3, seed=1)
        head.fit(x, y)
        out = head.predict_proba(x)
        assert out.shape == (16, 3)

    def test_predict_proba_returns_float32(self) -> None:
        head = SklearnGradientBoostingHead(
            n_labels=3, seed=0, max_iter=_FAST_MAX_ITER
        )
        x, y = _make_separable_xy(16, 4, 3, seed=2)
        head.fit(x, y)
        out = head.predict_proba(x)
        assert out.dtype == np.float32

    def test_predict_proba_in_unit_interval(self) -> None:
        head = SklearnGradientBoostingHead(
            n_labels=3, seed=0, max_iter=_FAST_MAX_ITER
        )
        x, y = _make_xy(16, 4, 3, seed=3)
        head.fit(x, y)
        out = head.predict_proba(x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0


# ---------------------------------------------------------------------------
# Determinism and reproducibility
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_heads_same_seed_same_data_byte_identical_output(self) -> None:
        x, y = _make_separable_xy(16, 4, 3, seed=4)
        a = SklearnGradientBoostingHead(
            n_labels=3, seed=42, max_iter=_FAST_MAX_ITER
        )
        b = SklearnGradientBoostingHead(
            n_labels=3, seed=42, max_iter=_FAST_MAX_ITER
        )
        a.fit(x, y)
        b.fit(x, y)
        out_a = a.predict_proba(x)
        out_b = b.predict_proba(x)
        np.testing.assert_array_equal(out_a, out_b)

    def test_repeated_predict_proba_is_byte_identical(self) -> None:
        x, y = _make_separable_xy(16, 4, 3, seed=5)
        head = SklearnGradientBoostingHead(
            n_labels=3, seed=7, max_iter=_FAST_MAX_ITER
        )
        head.fit(x, y)
        first = head.predict_proba(x)
        second = head.predict_proba(x)
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Multi-label independence and degenerate-fold fallbacks
# ---------------------------------------------------------------------------


class TestMultiLabelHandling:
    def test_all_zero_label_column_falls_back_to_empirical_rate(self) -> None:
        """A class with no positives in the training fold must not crash.

        ``HistGradientBoostingClassifier.fit`` raises when only one class is
        present. The adapter is documented to fall back to the empirical
        positive rate (which is 0.0 for an all-negative column).
        """
        x, y = _make_separable_xy(16, 4, 3, seed=6)
        # Force class 1 to be all-zero.
        y[:, 1] = 0
        head = SklearnGradientBoostingHead(
            n_labels=3, seed=0, max_iter=_FAST_MAX_ITER
        )
        head.fit(x, y)
        out = head.predict_proba(x)
        # Column 1 must be the empirical rate (0.0) for every row.
        np.testing.assert_array_equal(
            out[:, 1], np.zeros(out.shape[0], dtype=np.float32)
        )

    def test_all_one_label_column_falls_back_to_empirical_rate(self) -> None:
        x, y = _make_separable_xy(16, 4, 3, seed=7)
        # Force class 2 to be all-one.
        y[:, 2] = 1
        head = SklearnGradientBoostingHead(
            n_labels=3, seed=0, max_iter=_FAST_MAX_ITER
        )
        head.fit(x, y)
        out = head.predict_proba(x)
        np.testing.assert_array_equal(
            out[:, 2], np.ones(out.shape[0], dtype=np.float32)
        )

    def test_per_class_independence_changing_one_column_does_not_change_others(
        self,
    ) -> None:
        """Flipping labels in column 0 must not perturb columns 1 and 2.

        Each class is fit by its own ``HistGradientBoostingClassifier``, so
        no information ever crosses columns. We verify by training two heads
        on label matrices that differ only in column 0 and checking that
        columns 1 and 2 of ``predict_proba`` are byte-identical.
        """
        x, y = _make_separable_xy(16, 4, 3, seed=8)
        y_alt = y.copy()
        # Flip every label in column 0 of the alternate matrix.
        y_alt[:, 0] = 1 - y_alt[:, 0]

        head_a = SklearnGradientBoostingHead(
            n_labels=3, seed=11, max_iter=_FAST_MAX_ITER
        )
        head_b = SklearnGradientBoostingHead(
            n_labels=3, seed=11, max_iter=_FAST_MAX_ITER
        )
        head_a.fit(x, y)
        head_b.fit(x, y_alt)
        out_a = head_a.predict_proba(x)
        out_b = head_b.predict_proba(x)

        # Columns 1 and 2 are unchanged.
        np.testing.assert_array_equal(out_a[:, 1], out_b[:, 1])
        np.testing.assert_array_equal(out_a[:, 2], out_b[:, 2])


# ---------------------------------------------------------------------------
# Learning signal: separable data => positive class probabilities track labels
# ---------------------------------------------------------------------------


class TestLearningSignal:
    def test_predicted_probability_higher_for_positive_rows(self) -> None:
        """On near-separable data the GBT must rank positives above negatives."""
        x, y = _make_separable_xy(64, 4, 3, seed=9)
        head = SklearnGradientBoostingHead(n_labels=3, seed=13)
        head.fit(x, y)
        out = head.predict_proba(x)
        for j in range(3):
            pos_mask = y[:, j] == 1
            neg_mask = y[:, j] == 0
            if not pos_mask.any() or not neg_mask.any():
                continue
            mean_pos = float(out[pos_mask, j].mean())
            mean_neg = float(out[neg_mask, j].mean())
            assert mean_pos > mean_neg
