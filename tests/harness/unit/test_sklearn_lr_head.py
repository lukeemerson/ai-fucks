"""Unit tests for :class:`SklearnLogisticRegressionHead`.

Behavior assertions only: each test asserts on inputs/outputs through the
``ClassifierHeadPort`` surface (``fit`` / ``predict_proba``). No private
attribute access. Test data is intentionally tiny (16 rows, 4 features,
3 classes) so the entire file finishes in well under a second.

The headline behaviour under test is the rare-class rescue from
``class_weight='balanced'`` -- the load-bearing default that lifted macro-F1
from 0.094 (HGBT) to 0.157 in the TXRV-embedding ablation by recovering five
previously-zero classes.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.sklearn.lr_head import SklearnLogisticRegressionHead
from harness.domain.errors import ContractViolation


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
    """Build a linearly-separable multi-label dataset.

    Each class ``j`` fires when feature ``x[:, j % d]`` exceeds zero. LR
    can fit this exactly so the probability ranking is reliable.
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
            SklearnLogisticRegressionHead(n_labels=0, seed=0)

    def test_negative_n_labels_rejected(self) -> None:
        with pytest.raises(ContractViolation):
            SklearnLogisticRegressionHead(n_labels=-1, seed=0)

    def test_zero_max_iter_rejected(self) -> None:
        with pytest.raises(ContractViolation):
            SklearnLogisticRegressionHead(n_labels=3, seed=0, max_iter=0)

    def test_negative_max_iter_rejected(self) -> None:
        with pytest.raises(ContractViolation):
            SklearnLogisticRegressionHead(n_labels=3, seed=0, max_iter=-5)

    def test_zero_c_rejected(self) -> None:
        with pytest.raises(ContractViolation):
            SklearnLogisticRegressionHead(n_labels=3, seed=0, c_regularization=0.0)

    def test_negative_c_rejected(self) -> None:
        with pytest.raises(ContractViolation):
            SklearnLogisticRegressionHead(
                n_labels=3, seed=0, c_regularization=-1.0
            )


# ---------------------------------------------------------------------------
# Fit-input validation
# ---------------------------------------------------------------------------


class TestFitInputValidation:
    def test_features_must_be_2d(self) -> None:
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        x = np.zeros((10,), dtype=np.float32)
        y = np.zeros((10, 3), dtype=np.int8)
        with pytest.raises(ContractViolation):
            head.fit(x, y)

    def test_labels_must_be_2d(self) -> None:
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        x = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros((10,), dtype=np.int8)
        with pytest.raises(ContractViolation):
            head.fit(x, y)

    def test_row_mismatch_rejected(self) -> None:
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        x = np.zeros((8, 4), dtype=np.float32)
        y = np.zeros((10, 3), dtype=np.int8)
        with pytest.raises(ContractViolation):
            head.fit(x, y)

    def test_label_count_mismatch_rejected(self) -> None:
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        x = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros((10, 4), dtype=np.int8)  # 4 != n_labels=3
        with pytest.raises(ContractViolation):
            head.fit(x, y)


# ---------------------------------------------------------------------------
# Predict-input validation
# ---------------------------------------------------------------------------


class TestPredictInputValidation:
    def test_predict_before_fit_raises(self) -> None:
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        x = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ContractViolation):
            head.predict_proba(x)

    def test_predict_features_must_be_2d(self) -> None:
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
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
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        x, y = _make_separable_xy(16, 4, 3, seed=1)
        head.fit(x, y)
        out = head.predict_proba(x)
        assert out.shape == (16, 3)

    def test_predict_proba_returns_float32(self) -> None:
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        x, y = _make_separable_xy(16, 4, 3, seed=2)
        head.fit(x, y)
        out = head.predict_proba(x)
        assert out.dtype == np.float32

    def test_predict_proba_in_unit_interval(self) -> None:
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
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
        a = SklearnLogisticRegressionHead(n_labels=3, seed=42)
        b = SklearnLogisticRegressionHead(n_labels=3, seed=42)
        a.fit(x, y)
        b.fit(x, y)
        out_a = a.predict_proba(x)
        out_b = b.predict_proba(x)
        np.testing.assert_array_equal(out_a, out_b)

    def test_repeated_predict_proba_is_byte_identical(self) -> None:
        x, y = _make_separable_xy(16, 4, 3, seed=5)
        head = SklearnLogisticRegressionHead(n_labels=3, seed=7)
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

        ``LogisticRegression.fit`` raises when only one class is present.
        The adapter is documented to fall back to the empirical positive
        rate (which is 0.0 for an all-negative column).
        """
        x, y = _make_separable_xy(16, 4, 3, seed=6)
        y[:, 1] = 0
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        head.fit(x, y)
        out = head.predict_proba(x)
        np.testing.assert_array_equal(
            out[:, 1], np.zeros(out.shape[0], dtype=np.float32)
        )

    def test_all_one_label_column_falls_back_to_empirical_rate(self) -> None:
        x, y = _make_separable_xy(16, 4, 3, seed=7)
        y[:, 2] = 1
        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        head.fit(x, y)
        out = head.predict_proba(x)
        np.testing.assert_array_equal(
            out[:, 2], np.ones(out.shape[0], dtype=np.float32)
        )

    def test_per_class_independence_changing_one_column_does_not_change_others(
        self,
    ) -> None:
        """Flipping labels in column 0 must not perturb columns 1 and 2.

        Each class is fit by its own ``LogisticRegression``, so no
        information ever crosses columns. We verify by training two heads
        on label matrices that differ only in column 0 and checking that
        columns 1 and 2 of ``predict_proba`` are byte-identical.
        """
        x, y = _make_separable_xy(16, 4, 3, seed=8)
        y_alt = y.copy()
        y_alt[:, 0] = 1 - y_alt[:, 0]

        head_a = SklearnLogisticRegressionHead(n_labels=3, seed=11)
        head_b = SklearnLogisticRegressionHead(n_labels=3, seed=11)
        head_a.fit(x, y)
        head_b.fit(x, y_alt)
        out_a = head_a.predict_proba(x)
        out_b = head_b.predict_proba(x)

        np.testing.assert_array_equal(out_a[:, 1], out_b[:, 1])
        np.testing.assert_array_equal(out_a[:, 2], out_b[:, 2])


# ---------------------------------------------------------------------------
# Learning signal
# ---------------------------------------------------------------------------


class TestLearningSignal:
    def test_predicted_probability_higher_for_positive_rows(self) -> None:
        """On linearly separable data the LR head must rank positives above negatives."""
        x, y = _make_separable_xy(64, 4, 3, seed=9)
        head = SklearnLogisticRegressionHead(n_labels=3, seed=13)
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


# ---------------------------------------------------------------------------
# Class-weight balancing -- the load-bearing rare-class rescue
# ---------------------------------------------------------------------------


class TestClassWeightBalanced:
    def test_balanced_default_rescues_rare_class(self) -> None:
        """A 1%-positive column must produce non-degenerate probabilities.

        This is the headline behaviour from the TXRV-embedding ablation:
        ``class_weight='balanced'`` (the constructor default) prevents LR
        from collapsing to "always negative" on rare classes. Without it,
        five NIH-14 columns came back with all-zero predictions in the
        prior HGBT run.

        On a contrived 1%-prevalence column with a strong feature signal,
        the head's predicted probability for the few true-positive rows
        must be measurably above the all-negative floor.
        """
        rng = np.random.default_rng(20251201)
        n = 200
        d = 4
        # Class 0: 1% positives, perfectly correlated with feature 0.
        # Generate features first; pick the top 1% of feature 0 as positives.
        x = rng.standard_normal(size=(n, d)).astype(np.float32)
        n_pos = max(1, n // 100)
        # The n_pos rows with the largest feature-0 values are the positives.
        order = np.argsort(x[:, 0])[::-1]
        y = np.zeros((n, 3), dtype=np.int8)
        y[order[:n_pos], 0] = 1
        # Fill columns 1 and 2 with denser balanced-ish labels so the multi-
        # label structure is non-trivial.
        y[:, 1] = (x[:, 1] > 0).astype(np.int8)
        y[:, 2] = (x[:, 2] > 0).astype(np.int8)

        head = SklearnLogisticRegressionHead(n_labels=3, seed=0)
        head.fit(x, y)
        out = head.predict_proba(x)

        # Sanity: rare column produced varied probabilities (i.e. the head
        # didn't collapse to a single constant value across all rows).
        rare_probs = out[:, 0]
        assert float(rare_probs.std()) > 1e-3, (
            f"rare-class probabilities collapsed to near-constant: "
            f"std={float(rare_probs.std()):.6f}"
        )

        # Headline assertion: the true-positive rows should have a higher
        # mean probability than the all-negative floor. With class_weight=
        # 'balanced' this gap is large; without it LR tends to predict ~0
        # for everyone and the gap collapses.
        pos_mask = y[:, 0] == 1
        mean_pos = float(out[pos_mask, 0].mean())
        mean_neg = float(out[~pos_mask, 0].mean())
        assert mean_pos > mean_neg
        # Concrete floor: positives should exceed 0.5 (LR rates them as more
        # likely positive than negative). Without balancing this is not
        # guaranteed at 1% prevalence on a small sample.
        assert mean_pos > 0.5, (
            f"rare-class true positives not rescued: mean_pos={mean_pos:.4f}"
        )

    def test_unbalanced_option_does_not_rescue_rare_class(self) -> None:
        """Sanity check on the ``class_weight='none'`` opt-out.

        With ``class_weight='none'`` the rare-class probabilities collapse
        toward the empirical base rate, and the per-row mean predicted
        probability for rare-class positives drops below the balanced
        configuration. We don't assert a hard threshold (LR can still get
        lucky on some seeds) -- we only assert that the balanced default
        gives strictly higher mean-pos for the rare column.
        """
        rng = np.random.default_rng(20251202)
        n = 200
        d = 4
        x = rng.standard_normal(size=(n, d)).astype(np.float32)
        n_pos = max(1, n // 100)
        order = np.argsort(x[:, 0])[::-1]
        y = np.zeros((n, 1), dtype=np.int8)
        y[order[:n_pos], 0] = 1

        balanced = SklearnLogisticRegressionHead(
            n_labels=1, seed=0, class_weight="balanced"
        )
        unbalanced = SklearnLogisticRegressionHead(
            n_labels=1, seed=0, class_weight="none"
        )
        balanced.fit(x, y)
        unbalanced.fit(x, y)
        pos_mask = y[:, 0] == 1
        bal_mean_pos = float(balanced.predict_proba(x)[pos_mask, 0].mean())
        unbal_mean_pos = float(unbalanced.predict_proba(x)[pos_mask, 0].mean())
        # Balanced lifts the rare-class positive probability above unbalanced.
        assert bal_mean_pos > unbal_mean_pos
