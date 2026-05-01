"""Tests for ``harness.domain.types``.

These tests were written first (red) before any implementation existed.
They verify the dataclass contract from ARCHITECTURE.md section 3:

* every type is a frozen dataclass with slots,
* construction accepts the documented fields,
* invariants raise ``ContractViolation`` (or ``ConfigError`` for config),
* derived properties (e.g. ``MetricReport.macro_f1_mean``) are correct,
* ``ThresholdSet`` supports per-label lookup,
* equality is value based.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.domain.errors import ConfigError, ContractViolation
from harness.domain.types import (
    BootstrapConfig,
    Dataset,
    ExperimentConfig,
    ExperimentResult,
    MetricInterval,
    MetricReport,
    ModelCard,
    PerClassMetric,
    Predictions,
    Probabilities,
    Sample,
    Split,
    ThresholdConfig,
    ThresholdSet,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _proba_values(rows: int, cols: int, fill: float = 0.5) -> NDArray[np.float32]:
    return np.full((rows, cols), fill, dtype=np.float32)


def _pred_values(rows: int, cols: int) -> NDArray[np.int8]:
    return np.zeros((rows, cols), dtype=np.int8)


def _make_sample(idx: int = 0, n_labels: int = 2) -> Sample:
    return Sample(
        sample_id=f"s{idx}",
        patient_id=f"p{idx}",
        image_ref=f"img://{idx}",
        labels=tuple(0 for _ in range(n_labels)),
        metadata=MappingProxyType({"view": "PA"}),
    )


def _make_dataset(n_samples: int = 2, n_labels: int = 2) -> Dataset:
    return Dataset(
        name="fake",
        label_names=tuple(f"L{i}" for i in range(n_labels)),
        samples=tuple(_make_sample(i, n_labels) for i in range(n_samples)),
    )


def _make_metric_interval(point: float = 0.5) -> MetricInterval:
    return MetricInterval(point=point, lower=point - 0.05, upper=point + 0.05)


def _make_per_class(label: str, point: float = 0.5) -> PerClassMetric:
    return PerClassMetric(
        label=label,
        f1=_make_metric_interval(point),
        auroc=_make_metric_interval(point),
        auprc=_make_metric_interval(point),
        support=10,
    )


def _make_report(per_class: tuple[PerClassMetric, ...]) -> MetricReport:
    macro_point = sum(c.f1.point for c in per_class) / len(per_class)
    return MetricReport(
        macro_f1=_make_metric_interval(macro_point),
        macro_auroc=_make_metric_interval(macro_point),
        macro_auprc=_make_metric_interval(macro_point),
        per_class=per_class,
        n_bootstrap=100,
        seed=7,
    )


def _make_threshold_set(label_names: tuple[str, ...]) -> ThresholdSet:
    return ThresholdSet(
        label_names=label_names,
        thresholds=tuple(0.5 for _ in label_names),
        method="pr_sweep+shrink",
        shrinkage=0.1,
        clamp_lo=0.05,
        clamp_hi=0.95,
    )


def _make_bootstrap() -> BootstrapConfig:
    return BootstrapConfig(n_resamples=100, confidence=0.95, seed=7)


def _make_threshold_config() -> ThresholdConfig:
    return ThresholdConfig(method="pr_sweep", shrinkage=0.1, clamp_lo=0.05, clamp_hi=0.95)


def _make_experiment_config(label_names: tuple[str, ...] = ("L0", "L1")) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="exp",
        dataset_name="fake",
        label_names=label_names,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
        bootstrap=_make_bootstrap(),
        threshold=_make_threshold_config(),
        backbone_id="hash-fake",
        head_id="logreg",
        calibrator_id="isotonic",
        artifact_root="mem://run",
        notes="",
    )


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


class TestSample:
    def test_construct(self) -> None:
        sample = _make_sample(idx=1, n_labels=3)
        assert sample.sample_id == "s1"
        assert sample.patient_id == "p1"
        assert sample.labels == (0, 0, 0)
        assert sample.metadata["view"] == "PA"

    def test_frozen(self) -> None:
        sample = _make_sample()
        with pytest.raises(FrozenInstanceError):
            sample.sample_id = "other"  # type: ignore[misc]  # frozen dataclass

    def test_equality(self) -> None:
        assert _make_sample(0, 2) == _make_sample(0, 2)
        assert _make_sample(0, 2) != _make_sample(1, 2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_construct(self) -> None:
        ds = _make_dataset(3, 2)
        assert ds.name == "fake"
        assert ds.label_names == ("L0", "L1")
        assert len(ds.samples) == 3

    def test_frozen(self) -> None:
        ds = _make_dataset()
        with pytest.raises(FrozenInstanceError):
            ds.name = "other"  # type: ignore[misc]  # frozen dataclass

    def test_rejects_label_length_mismatch(self) -> None:
        bad_sample = Sample(
            sample_id="s0",
            patient_id="p0",
            image_ref="img://0",
            labels=(0, 1, 0),  # length 3
            metadata=MappingProxyType({}),
        )
        with pytest.raises(ContractViolation):
            Dataset(name="fake", label_names=("L0", "L1"), samples=(bad_sample,))

    def test_equality(self) -> None:
        assert _make_dataset(2, 2) == _make_dataset(2, 2)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_construct(self) -> None:
        split = Split(train_indices=(0, 1), val_indices=(2,), test_indices=(3,), seed=7)
        assert split.train_indices == (0, 1)
        assert split.seed == 7

    def test_frozen(self) -> None:
        split = Split(train_indices=(0,), val_indices=(1,), test_indices=(2,), seed=1)
        with pytest.raises(FrozenInstanceError):
            split.seed = 2  # type: ignore[misc]  # frozen dataclass

    def test_rejects_overlapping_indices(self) -> None:
        with pytest.raises(ContractViolation):
            Split(train_indices=(0, 1), val_indices=(1,), test_indices=(2,), seed=1)

    def test_rejects_negative_indices(self) -> None:
        with pytest.raises(ContractViolation):
            Split(train_indices=(-1,), val_indices=(0,), test_indices=(1,), seed=1)

    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(ContractViolation):
            Split(train_indices=(0,), val_indices=(1,), test_indices=(2,), seed=-1)


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------


class TestProbabilities:
    def test_construct(self) -> None:
        p = Probabilities(
            sample_ids=("s0", "s1"),
            label_names=("L0", "L1"),
            values=_proba_values(2, 2, 0.3),
        )
        assert p.sample_ids == ("s0", "s1")
        assert p.values.shape == (2, 2)
        assert p.values.dtype == np.float32

    def test_frozen(self) -> None:
        p = Probabilities(
            sample_ids=("s0",),
            label_names=("L0",),
            values=_proba_values(1, 1, 0.5),
        )
        with pytest.raises(FrozenInstanceError):
            p.sample_ids = ()  # type: ignore[misc]  # frozen dataclass

    def test_rejects_values_outside_unit_interval(self) -> None:
        bad = np.array([[1.5, 0.4]], dtype=np.float32)
        with pytest.raises(ContractViolation):
            Probabilities(sample_ids=("s0",), label_names=("L0", "L1"), values=bad)

    def test_rejects_negative_values(self) -> None:
        bad = np.array([[-0.1, 0.4]], dtype=np.float32)
        with pytest.raises(ContractViolation):
            Probabilities(sample_ids=("s0",), label_names=("L0", "L1"), values=bad)

    def test_rejects_row_count_mismatch(self) -> None:
        with pytest.raises(ContractViolation):
            Probabilities(
                sample_ids=("s0", "s1"),
                label_names=("L0",),
                values=_proba_values(1, 1, 0.5),
            )

    def test_rejects_column_count_mismatch(self) -> None:
        with pytest.raises(ContractViolation):
            Probabilities(
                sample_ids=("s0",),
                label_names=("L0", "L1"),
                values=_proba_values(1, 1, 0.5),
            )

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ContractViolation):
            Probabilities(
                sample_ids=("s0",),
                label_names=("L0",),
                values=np.array([0.5], dtype=np.float32),
            )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class TestPredictions:
    def test_construct(self) -> None:
        preds = Predictions(
            sample_ids=("s0",),
            label_names=("L0", "L1"),
            values=np.array([[0, 1]], dtype=np.int8),
        )
        assert preds.values.tolist() == [[0, 1]]

    def test_frozen(self) -> None:
        preds = Predictions(
            sample_ids=("s0",),
            label_names=("L0",),
            values=_pred_values(1, 1),
        )
        with pytest.raises(FrozenInstanceError):
            preds.sample_ids = ()  # type: ignore[misc]  # frozen dataclass

    def test_rejects_non_binary_values(self) -> None:
        bad = np.array([[2, 0]], dtype=np.int8)
        with pytest.raises(ContractViolation):
            Predictions(sample_ids=("s0",), label_names=("L0", "L1"), values=bad)

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ContractViolation):
            Predictions(
                sample_ids=("s0", "s1"),
                label_names=("L0",),
                values=_pred_values(1, 1),
            )


# ---------------------------------------------------------------------------
# MetricInterval / PerClassMetric / MetricReport
# ---------------------------------------------------------------------------


class TestMetricInterval:
    def test_construct(self) -> None:
        mi = MetricInterval(point=0.5, lower=0.4, upper=0.6)
        assert mi.point == 0.5

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ContractViolation):
            MetricInterval(point=0.5, lower=0.6, upper=0.4)

    def test_accepts_point_outside_bounds_for_degenerate_cases(self) -> None:
        # Wave 6a relaxed the invariant: degenerate bootstrap distributions
        # (e.g., constant resamples on tiny supports) can leave the point
        # estimate outside the [lower, upper] CI. Construction must succeed.
        mi = MetricInterval(point=0.9, lower=0.0, upper=0.5)
        assert mi.point == 0.9
        assert mi.lower == 0.0
        assert mi.upper == 0.5


class TestPerClassMetric:
    def test_construct(self) -> None:
        m = _make_per_class("L0", 0.7)
        assert m.label == "L0"
        assert m.support == 10

    def test_rejects_negative_support(self) -> None:
        with pytest.raises(ContractViolation):
            PerClassMetric(
                label="L0",
                f1=_make_metric_interval(),
                auroc=_make_metric_interval(),
                auprc=_make_metric_interval(),
                support=-1,
            )


class TestMetricReport:
    def test_construct(self) -> None:
        per_class = (_make_per_class("L0", 0.6), _make_per_class("L1", 0.8))
        report = _make_report(per_class)
        assert len(report.per_class) == 2
        assert report.n_bootstrap == 100

    def test_macro_f1_mean_matches_per_class_average(self) -> None:
        per_class = (_make_per_class("L0", 0.6), _make_per_class("L1", 0.8))
        report = _make_report(per_class)
        assert report.macro_f1_mean == pytest.approx(0.7)

    def test_frozen(self) -> None:
        per_class = (_make_per_class("L0"),)
        report = _make_report(per_class)
        with pytest.raises(FrozenInstanceError):
            report.n_bootstrap = 99  # type: ignore[misc]  # frozen dataclass

    def test_rejects_empty_per_class(self) -> None:
        with pytest.raises(ContractViolation):
            MetricReport(
                macro_f1=_make_metric_interval(),
                macro_auroc=_make_metric_interval(),
                macro_auprc=_make_metric_interval(),
                per_class=(),
                n_bootstrap=100,
                seed=7,
            )

    def test_rejects_non_positive_n_bootstrap(self) -> None:
        with pytest.raises(ContractViolation):
            MetricReport(
                macro_f1=_make_metric_interval(),
                macro_auroc=_make_metric_interval(),
                macro_auprc=_make_metric_interval(),
                per_class=(_make_per_class("L0"),),
                n_bootstrap=0,
                seed=7,
            )


# ---------------------------------------------------------------------------
# ThresholdSet
# ---------------------------------------------------------------------------


class TestThresholdSet:
    def test_construct(self) -> None:
        ts = _make_threshold_set(("L0", "L1"))
        assert ts.thresholds == (0.5, 0.5)

    def test_frozen(self) -> None:
        ts = _make_threshold_set(("L0",))
        with pytest.raises(FrozenInstanceError):
            ts.method = "other"  # type: ignore[misc]  # frozen dataclass

    def test_lookup_by_label(self) -> None:
        ts = ThresholdSet(
            label_names=("L0", "L1"),
            thresholds=(0.3, 0.7),
            method="pr_sweep+shrink",
            shrinkage=0.0,
            clamp_lo=0.0,
            clamp_hi=1.0,
        )
        assert ts.threshold_for("L0") == pytest.approx(0.3)
        assert ts.threshold_for("L1") == pytest.approx(0.7)

    def test_lookup_unknown_label_raises(self) -> None:
        ts = _make_threshold_set(("L0",))
        with pytest.raises(KeyError):
            ts.threshold_for("nope")

    def test_iteration_yields_label_threshold_pairs(self) -> None:
        ts = ThresholdSet(
            label_names=("L0", "L1"),
            thresholds=(0.3, 0.7),
            method="pr_sweep+shrink",
            shrinkage=0.0,
            clamp_lo=0.0,
            clamp_hi=1.0,
        )
        assert list(ts) == [("L0", 0.3), ("L1", 0.7)]

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ContractViolation):
            ThresholdSet(
                label_names=("L0", "L1"),
                thresholds=(0.5,),
                method="pr_sweep+shrink",
                shrinkage=0.1,
                clamp_lo=0.0,
                clamp_hi=1.0,
            )

    def test_rejects_thresholds_outside_clamp(self) -> None:
        with pytest.raises(ContractViolation):
            ThresholdSet(
                label_names=("L0",),
                thresholds=(0.99,),
                method="pr_sweep+shrink",
                shrinkage=0.1,
                clamp_lo=0.1,
                clamp_hi=0.9,
            )

    def test_rejects_clamp_outside_unit_interval(self) -> None:
        with pytest.raises(ContractViolation):
            ThresholdSet(
                label_names=("L0",),
                thresholds=(0.5,),
                method="pr_sweep+shrink",
                shrinkage=0.1,
                clamp_lo=-0.1,
                clamp_hi=0.9,
            )

    def test_rejects_inverted_clamp(self) -> None:
        with pytest.raises(ContractViolation):
            ThresholdSet(
                label_names=("L0",),
                thresholds=(0.5,),
                method="pr_sweep+shrink",
                shrinkage=0.1,
                clamp_lo=0.9,
                clamp_hi=0.1,
            )

    def test_rejects_shrinkage_outside_unit_interval(self) -> None:
        with pytest.raises(ContractViolation):
            ThresholdSet(
                label_names=("L0",),
                thresholds=(0.5,),
                method="pr_sweep+shrink",
                shrinkage=1.5,
                clamp_lo=0.0,
                clamp_hi=1.0,
            )


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------


class TestModelCard:
    def test_construct(self) -> None:
        per_class = (_make_per_class("L0"), _make_per_class("L1"))
        report = _make_report(per_class)
        card = ModelCard(
            name="exp",
            version="0.1",
            created_at=datetime(2026, 5, 1, tzinfo=UTC),
            backbone="hash-fake",
            head="logreg",
            calibrator="isotonic",
            threshold_method="pr_sweep+shrink",
            label_names=("L0", "L1"),
            train_size=100,
            val_size=20,
            test_size=20,
            config_hash="deadbeef",
            metrics=report,
            notes="",
        )
        assert card.name == "exp"
        assert card.metrics is report

    def test_frozen(self) -> None:
        per_class = (_make_per_class("L0"),)
        card = ModelCard(
            name="exp",
            version="0.1",
            created_at=datetime(2026, 5, 1, tzinfo=UTC),
            backbone="hash-fake",
            head="logreg",
            calibrator="isotonic",
            threshold_method="pr_sweep+shrink",
            label_names=("L0",),
            train_size=10,
            val_size=2,
            test_size=2,
            config_hash="x",
            metrics=_make_report(per_class),
            notes="",
        )
        with pytest.raises(FrozenInstanceError):
            card.name = "other"  # type: ignore[misc]  # frozen dataclass


# ---------------------------------------------------------------------------
# BootstrapConfig / ThresholdConfig / ExperimentConfig
# ---------------------------------------------------------------------------


class TestBootstrapConfig:
    def test_construct(self) -> None:
        b = _make_bootstrap()
        assert b.n_resamples == 100

    def test_rejects_non_positive_resamples(self) -> None:
        with pytest.raises(ConfigError):
            BootstrapConfig(n_resamples=0, confidence=0.95, seed=7)

    def test_rejects_confidence_outside_open_unit_interval(self) -> None:
        with pytest.raises(ConfigError):
            BootstrapConfig(n_resamples=10, confidence=1.5, seed=7)
        with pytest.raises(ConfigError):
            BootstrapConfig(n_resamples=10, confidence=0.0, seed=7)

    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(ConfigError):
            BootstrapConfig(n_resamples=10, confidence=0.95, seed=-1)


class TestThresholdConfig:
    def test_construct(self) -> None:
        c = _make_threshold_config()
        assert c.method == "pr_sweep"

    def test_rejects_shrinkage_outside_unit_interval(self) -> None:
        with pytest.raises(ConfigError):
            ThresholdConfig(method="pr_sweep", shrinkage=1.5, clamp_lo=0.0, clamp_hi=1.0)

    def test_rejects_inverted_clamp(self) -> None:
        with pytest.raises(ConfigError):
            ThresholdConfig(method="pr_sweep", shrinkage=0.1, clamp_lo=0.9, clamp_hi=0.1)

    def test_rejects_clamp_outside_unit_interval(self) -> None:
        with pytest.raises(ConfigError):
            ThresholdConfig(method="pr_sweep", shrinkage=0.1, clamp_lo=-0.1, clamp_hi=1.0)


class TestExperimentConfig:
    def test_construct(self) -> None:
        cfg = _make_experiment_config()
        assert cfg.experiment_name == "exp"
        assert cfg.label_names == ("L0", "L1")

    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(ConfigError):
            ExperimentConfig(
                experiment_name="exp",
                dataset_name="fake",
                label_names=("L0",),
                val_fraction=0.2,
                test_fraction=0.2,
                seed=-1,
                bootstrap=_make_bootstrap(),
                threshold=_make_threshold_config(),
                backbone_id="x",
                head_id="x",
                calibrator_id="x",
                artifact_root="mem://",
                notes="",
            )

    def test_rejects_empty_label_names(self) -> None:
        with pytest.raises(ConfigError):
            ExperimentConfig(
                experiment_name="exp",
                dataset_name="fake",
                label_names=(),
                val_fraction=0.2,
                test_fraction=0.2,
                seed=7,
                bootstrap=_make_bootstrap(),
                threshold=_make_threshold_config(),
                backbone_id="x",
                head_id="x",
                calibrator_id="x",
                artifact_root="mem://",
                notes="",
            )

    def test_rejects_fractions_summing_to_or_above_one(self) -> None:
        with pytest.raises(ConfigError):
            ExperimentConfig(
                experiment_name="exp",
                dataset_name="fake",
                label_names=("L0",),
                val_fraction=0.6,
                test_fraction=0.5,
                seed=7,
                bootstrap=_make_bootstrap(),
                threshold=_make_threshold_config(),
                backbone_id="x",
                head_id="x",
                calibrator_id="x",
                artifact_root="mem://",
                notes="",
            )

    def test_rejects_negative_fractions(self) -> None:
        with pytest.raises(ConfigError):
            ExperimentConfig(
                experiment_name="exp",
                dataset_name="fake",
                label_names=("L0",),
                val_fraction=-0.1,
                test_fraction=0.2,
                seed=7,
                bootstrap=_make_bootstrap(),
                threshold=_make_threshold_config(),
                backbone_id="x",
                head_id="x",
                calibrator_id="x",
                artifact_root="mem://",
                notes="",
            )

    def test_rejects_empty_experiment_name(self) -> None:
        with pytest.raises(ConfigError):
            ExperimentConfig(
                experiment_name="",
                dataset_name="fake",
                label_names=("L0",),
                val_fraction=0.2,
                test_fraction=0.2,
                seed=7,
                bootstrap=_make_bootstrap(),
                threshold=_make_threshold_config(),
                backbone_id="x",
                head_id="x",
                calibrator_id="x",
                artifact_root="mem://",
                notes="",
            )

    def test_equality(self) -> None:
        assert _make_experiment_config() == _make_experiment_config()


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult:
    def _build(self) -> ExperimentResult:
        cfg = _make_experiment_config(("L0", "L1"))
        split = Split(train_indices=(0,), val_indices=(1,), test_indices=(2,), seed=7)
        ts = _make_threshold_set(("L0", "L1"))
        val_p = Probabilities(
            sample_ids=("s0",),
            label_names=("L0", "L1"),
            values=_proba_values(1, 2, 0.5),
        )
        test_p = Probabilities(
            sample_ids=("s1",),
            label_names=("L0", "L1"),
            values=_proba_values(1, 2, 0.5),
        )
        test_preds = Predictions(
            sample_ids=("s1",),
            label_names=("L0", "L1"),
            values=np.zeros((1, 2), dtype=np.int8),
        )
        per_class = (_make_per_class("L0"), _make_per_class("L1"))
        report = _make_report(per_class)
        card = ModelCard(
            name="exp",
            version="0.1",
            created_at=datetime(2026, 5, 1, tzinfo=UTC),
            backbone="hash-fake",
            head="logreg",
            calibrator="isotonic",
            threshold_method="pr_sweep+shrink",
            label_names=("L0", "L1"),
            train_size=1,
            val_size=1,
            test_size=1,
            config_hash="x",
            metrics=report,
            notes="",
        )
        return ExperimentResult(
            config=cfg,
            split=split,
            thresholds=ts,
            val_probabilities=val_p,
            test_probabilities=test_p,
            test_predictions=test_preds,
            report=report,
            model_card=card,
            artifact_uris=MappingProxyType({"model_card": "mem://card"}),
        )

    def test_construct(self) -> None:
        result = self._build()
        assert result.config.experiment_name == "exp"
        assert result.artifact_uris["model_card"] == "mem://card"

    def test_frozen(self) -> None:
        result = self._build()
        with pytest.raises(FrozenInstanceError):
            result.config = result.config  # type: ignore[misc]  # frozen dataclass
