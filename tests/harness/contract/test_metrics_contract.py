"""Contract tests for :class:`harness.ports.metrics.MetricsPort`.

Asserts *behavior* (shape, invariants, determinism, error funneling).
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.adapters.fakes.metrics import CountingFakeMetrics
from harness.adapters.sklearn.metrics import BootstrapMetrics
from harness.domain.errors import HarnessError
from harness.domain.types import (
    BootstrapConfig,
    MetricInterval,
    MetricReport,
    PerClassMetric,
    Probabilities,
    ThresholdSet,
)
from harness.ports.metrics import MetricsPort


def _probs(
    n_samples: int = 8,
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


def _labels(n_samples: int = 8, n_labels: int = 3) -> list[list[int]]:
    return [[(i + j) % 2 for j in range(n_labels)] for i in range(n_samples)]


def _thresholds(label_names: tuple[str, ...]) -> ThresholdSet:
    return ThresholdSet(
        label_names=label_names,
        thresholds=tuple(0.5 for _ in label_names),
        method="manual",
        shrinkage=0.0,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )


def _bootstrap(seed: int = 7) -> BootstrapConfig:
    return BootstrapConfig(n_resamples=16, confidence=0.95, seed=seed)


class MetricsPortContract:
    @pytest.fixture
    def adapter(self) -> MetricsPort:
        raise NotImplementedError

    def test_evaluate_returns_metric_report(self, adapter: MetricsPort) -> None:
        probs = _probs()
        report = adapter.evaluate(
            probs,
            _labels(),
            _thresholds(probs.label_names),
            bootstrap=_bootstrap(),
        )
        assert isinstance(report, MetricReport)

    def test_evaluate_per_class_matches_label_names(
        self, adapter: MetricsPort
    ) -> None:
        probs = _probs(n_samples=10, n_labels=4)
        report = adapter.evaluate(
            probs,
            _labels(n_samples=10, n_labels=4),
            _thresholds(probs.label_names),
            bootstrap=_bootstrap(),
        )
        assert len(report.per_class) == len(probs.label_names)
        assert tuple(c.label for c in report.per_class) == probs.label_names

    def test_evaluate_intervals_are_bracketed(
        self, adapter: MetricsPort
    ) -> None:
        probs = _probs()
        report = adapter.evaluate(
            probs,
            _labels(),
            _thresholds(probs.label_names),
            bootstrap=_bootstrap(),
        )
        for interval in (
            report.macro_f1,
            report.macro_auroc,
            report.macro_auprc,
        ):
            assert isinstance(interval, MetricInterval)
            assert interval.lower <= interval.point <= interval.upper
        for cls in report.per_class:
            assert isinstance(cls, PerClassMetric)
            for interval in (cls.f1, cls.auroc, cls.auprc):
                assert interval.lower <= interval.point <= interval.upper

    def test_evaluate_macro_f1_is_mean_of_per_class_f1(
        self, adapter: MetricsPort
    ) -> None:
        probs = _probs(n_samples=12, n_labels=3, seed=2)
        report = adapter.evaluate(
            probs,
            _labels(n_samples=12, n_labels=3),
            _thresholds(probs.label_names),
            bootstrap=_bootstrap(),
        )
        expected = sum(c.f1.point for c in report.per_class) / len(
            report.per_class
        )
        assert report.macro_f1.point == pytest.approx(expected, abs=1e-6)

    def test_evaluate_is_deterministic_given_same_seed(
        self, adapter: MetricsPort
    ) -> None:
        probs = _probs(n_samples=10, n_labels=3, seed=3)
        labels = _labels(n_samples=10, n_labels=3)
        ts = _thresholds(probs.label_names)
        cfg = _bootstrap(seed=42)
        a = adapter.evaluate(probs, labels, ts, bootstrap=cfg)
        b = adapter.evaluate(probs, labels, ts, bootstrap=cfg)
        assert a.macro_f1.point == b.macro_f1.point
        assert a.macro_f1.lower == b.macro_f1.lower
        assert a.macro_f1.upper == b.macro_f1.upper
        for ca, cb in zip(a.per_class, b.per_class, strict=True):
            assert ca.f1.point == cb.f1.point
            assert ca.f1.lower == cb.f1.lower
            assert ca.f1.upper == cb.f1.upper

    def test_evaluate_supports_match_label_count(
        self, adapter: MetricsPort
    ) -> None:
        probs = _probs(n_samples=6, n_labels=3)
        labels = _labels(n_samples=6, n_labels=3)
        report = adapter.evaluate(
            probs, labels, _thresholds(probs.label_names), bootstrap=_bootstrap()
        )
        # support is the number of positives per class; bound by n_samples.
        for cls in report.per_class:
            assert 0 <= cls.support <= 6

    def test_evaluate_rejects_shape_mismatch_labels(
        self, adapter: MetricsPort
    ) -> None:
        probs = _probs(n_samples=5, n_labels=3)
        # Wrong row count.
        bad_labels = _labels(n_samples=4, n_labels=3)
        with pytest.raises(HarnessError):
            adapter.evaluate(
                probs,
                bad_labels,
                _thresholds(probs.label_names),
                bootstrap=_bootstrap(),
            )

    def test_evaluate_rejects_shape_mismatch_thresholds(
        self, adapter: MetricsPort
    ) -> None:
        probs = _probs(n_samples=5, n_labels=3)
        # ThresholdSet whose label_names disagree with probabilities.
        bad_ts = ThresholdSet(
            label_names=("a", "b"),
            thresholds=(0.5, 0.5),
            method="manual",
            shrinkage=0.0,
            clamp_lo=0.0,
            clamp_hi=1.0,
        )
        with pytest.raises(HarnessError):
            adapter.evaluate(
                probs,
                _labels(n_samples=5, n_labels=3),
                bad_ts,
                bootstrap=_bootstrap(),
            )


class TestCountingFakeMetricsContract(MetricsPortContract):
    @pytest.fixture
    def adapter(self) -> MetricsPort:
        return CountingFakeMetrics()


class TestBootstrapMetricsContract(MetricsPortContract):
    @pytest.fixture
    def adapter(self) -> MetricsPort:
        return BootstrapMetrics()
