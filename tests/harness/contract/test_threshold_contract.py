"""Contract tests for :class:`harness.ports.threshold.ThresholdPort`.

Per ARCHITECTURE.md section 7.1 every port has an abstract contract test class
asserting *behavior* (shape, invariants, determinism). Concrete adapters
subclass and supply an ``adapter`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.adapters.fakes.threshold import FixedFakeThreshold
from harness.adapters.sklearn.threshold import PrSweepShrinkageThreshold
from harness.domain.errors import HarnessError
from harness.domain.types import (
    Predictions,
    Probabilities,
    ThresholdConfig,
    ThresholdSet,
)
from harness.ports.threshold import ThresholdPort


def _probs(
    n_samples: int = 6,
    n_labels: int = 3,
    *,
    seed: int = 0,
) -> Probabilities:
    rng = np.random.default_rng(seed)
    values = rng.random((n_samples, n_labels)).astype(np.float32)
    return Probabilities(
        sample_ids=tuple(f"s{i}" for i in range(n_samples)),
        label_names=tuple(f"l{j}" for j in range(n_labels)),
        values=values,
    )


def _labels(n_samples: int = 6, n_labels: int = 3) -> list[list[int]]:
    return [[(i + j) % 2 for j in range(n_labels)] for i in range(n_samples)]


def _config() -> ThresholdConfig:
    return ThresholdConfig(
        method="pr_sweep",
        shrinkage=0.0,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )


class ThresholdPortContract:
    """Abstract contract; subclasses provide an ``adapter`` fixture."""

    @pytest.fixture
    def adapter(self) -> ThresholdPort:
        raise NotImplementedError

    def test_identifier_is_non_empty_string(
        self, adapter: ThresholdPort
    ) -> None:
        assert isinstance(adapter.identifier, str)
        assert adapter.identifier

    def test_fit_returns_threshold_set_with_one_threshold_per_label(
        self, adapter: ThresholdPort
    ) -> None:
        probs = _probs(n_samples=8, n_labels=4)
        labels = _labels(n_samples=8, n_labels=4)
        ts = adapter.fit(probs, labels, config=_config())
        assert isinstance(ts, ThresholdSet)
        assert len(ts.thresholds) == len(probs.label_names)
        assert ts.label_names == probs.label_names

    def test_fit_thresholds_are_in_unit_interval(
        self, adapter: ThresholdPort
    ) -> None:
        probs = _probs()
        ts = adapter.fit(probs, _labels(), config=_config())
        for t in ts.thresholds:
            assert 0.0 <= t <= 1.0

    def test_fit_respects_clamp_bounds(self, adapter: ThresholdPort) -> None:
        cfg = ThresholdConfig(
            method="pr_sweep",
            shrinkage=0.0,
            clamp_lo=0.3,
            clamp_hi=0.7,
        )
        ts = adapter.fit(_probs(), _labels(), config=cfg)
        for t in ts.thresholds:
            assert cfg.clamp_lo <= t <= cfg.clamp_hi

    def test_apply_returns_predictions_of_correct_shape(
        self, adapter: ThresholdPort
    ) -> None:
        probs = _probs(n_samples=5, n_labels=3)
        ts = adapter.fit(probs, _labels(n_samples=5, n_labels=3), config=_config())
        preds = adapter.apply(probs, ts)
        assert isinstance(preds, Predictions)
        assert preds.values.shape == (5, 3)
        assert preds.sample_ids == probs.sample_ids
        assert preds.label_names == probs.label_names

    def test_apply_produces_zero_or_one_only(
        self, adapter: ThresholdPort
    ) -> None:
        probs = _probs()
        ts = adapter.fit(probs, _labels(), config=_config())
        preds = adapter.apply(probs, ts)
        unique = {int(v) for v in np.unique(preds.values).tolist()}
        assert unique.issubset({0, 1})

    def test_apply_is_deterministic(self, adapter: ThresholdPort) -> None:
        probs = _probs()
        ts = adapter.fit(probs, _labels(), config=_config())
        a = adapter.apply(probs, ts)
        b = adapter.apply(probs, ts)
        assert np.array_equal(a.values, b.values)

    def test_apply_thresholds_per_class_independence(
        self, adapter: ThresholdPort
    ) -> None:
        """Changing the threshold for label k must not change label j's preds."""
        probs = _probs(n_samples=10, n_labels=3, seed=1)
        labels = _labels(n_samples=10, n_labels=3)
        ts = adapter.fit(probs, labels, config=_config())
        # Build an alternate set with label-0 threshold flipped to the opposite
        # extreme; label-1 and label-2 thresholds unchanged.
        original = list(ts.thresholds)
        new_thresh = 0.99 if original[0] < 0.5 else 0.01
        alt = ThresholdSet(
            label_names=ts.label_names,
            thresholds=(new_thresh, original[1], original[2]),
            method=ts.method,
            shrinkage=ts.shrinkage,
            clamp_lo=0.0,
            clamp_hi=1.0,
        )
        a = adapter.apply(probs, ts)
        b = adapter.apply(probs, alt)
        # Columns 1 and 2 must be unchanged; column 0 may differ.
        assert np.array_equal(a.values[:, 1], b.values[:, 1])
        assert np.array_equal(a.values[:, 2], b.values[:, 2])

    def test_apply_rejects_shape_mismatch(self, adapter: ThresholdPort) -> None:
        probs = _probs(n_samples=4, n_labels=3)
        # Build a ThresholdSet with the *wrong* number of label_names.
        bad_ts = ThresholdSet(
            label_names=("only_one",),
            thresholds=(0.5,),
            method="manual",
            shrinkage=0.0,
            clamp_lo=0.0,
            clamp_hi=1.0,
        )
        with pytest.raises(HarnessError):
            adapter.apply(probs, bad_ts)

    def test_fit_is_deterministic_for_same_inputs(
        self, adapter: ThresholdPort
    ) -> None:
        probs = _probs()
        labels = _labels()
        a = adapter.fit(probs, labels, config=_config())
        b = adapter.fit(probs, labels, config=_config())
        assert a.thresholds == b.thresholds


class TestFixedFakeThresholdContract(ThresholdPortContract):
    @pytest.fixture
    def adapter(self) -> ThresholdPort:
        return FixedFakeThreshold()


class TestPrSweepShrinkageThresholdContract(ThresholdPortContract):
    @pytest.fixture
    def adapter(self) -> ThresholdPort:
        return PrSweepShrinkageThreshold()
