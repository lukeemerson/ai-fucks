"""Contract tests for :class:`harness.ports.artifact_store.ArtifactStorePort`.

Asserts the four required write methods return non-empty path strings, that
the in-memory fake's inspection surface returns the written payload, and that
re-writing the same logical name is idempotent (returns the same path).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from harness.adapters.fakes.artifact_store import InMemoryFakeArtifactStore
from harness.adapters.fs.artifact_store import FilesystemArtifactStore
from harness.domain.types import (
    MetricInterval,
    MetricReport,
    ModelCard,
    PerClassMetric,
    Predictions,
    ThresholdSet,
)
from harness.ports.artifact_store import ArtifactStorePort


def _model_card() -> ModelCard:
    interval = MetricInterval(point=0.5, lower=0.5, upper=0.5)
    report = MetricReport(
        macro_f1=interval,
        macro_auroc=interval,
        macro_auprc=interval,
        per_class=(
            PerClassMetric(
                label="a",
                f1=interval,
                auroc=interval,
                auprc=interval,
                support=1,
            ),
        ),
        n_bootstrap=1,
        seed=0,
    )
    return ModelCard(
        name="m",
        version="v1",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        backbone="b",
        head="h",
        calibrator="c",
        threshold_method="t",
        label_names=("a",),
        train_size=1,
        val_size=1,
        test_size=1,
        config_hash="abc",
        metrics=report,
        notes="n",
    )


def _predictions() -> Predictions:
    return Predictions(
        sample_ids=("s0", "s1"),
        label_names=("a", "b"),
        values=np.array([[0, 1], [1, 0]], dtype=np.int8),
    )


def _thresholds() -> ThresholdSet:
    return ThresholdSet(
        label_names=("a", "b"),
        thresholds=(0.4, 0.6),
        method="manual",
        shrinkage=0.0,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )


def _metric_report() -> MetricReport:
    interval = MetricInterval(point=0.7, lower=0.6, upper=0.8)
    return MetricReport(
        macro_f1=interval,
        macro_auroc=interval,
        macro_auprc=interval,
        per_class=(
            PerClassMetric(
                label="a",
                f1=interval,
                auroc=interval,
                auprc=interval,
                support=2,
            ),
        ),
        n_bootstrap=8,
        seed=0,
    )


class ArtifactStorePortContract:
    @pytest.fixture
    def adapter(self) -> ArtifactStorePort:
        raise NotImplementedError

    def test_write_model_card_returns_non_empty_path(
        self, adapter: ArtifactStorePort
    ) -> None:
        path = adapter.write_model_card(_model_card())
        assert isinstance(path, str)
        assert path

    def test_write_predictions_returns_non_empty_path(
        self, adapter: ArtifactStorePort
    ) -> None:
        path = adapter.write_predictions(_predictions(), "test")
        assert isinstance(path, str)
        assert path

    def test_write_thresholds_returns_non_empty_path(
        self, adapter: ArtifactStorePort
    ) -> None:
        path = adapter.write_thresholds(_thresholds())
        assert isinstance(path, str)
        assert path

    def test_write_metric_report_returns_non_empty_path(
        self, adapter: ArtifactStorePort
    ) -> None:
        path = adapter.write_metric_report(_metric_report())
        assert isinstance(path, str)
        assert path

    def test_paths_distinguish_artifact_kinds(
        self, adapter: ArtifactStorePort
    ) -> None:
        card_path = adapter.write_model_card(_model_card())
        pred_path = adapter.write_predictions(_predictions(), "test")
        thr_path = adapter.write_thresholds(_thresholds())
        rep_path = adapter.write_metric_report(_metric_report())
        assert len({card_path, pred_path, thr_path, rep_path}) == 4

    def test_write_predictions_distinguishes_names(
        self, adapter: ArtifactStorePort
    ) -> None:
        a = adapter.write_predictions(_predictions(), "val")
        b = adapter.write_predictions(_predictions(), "test")
        assert a != b

    def test_write_predictions_idempotent_for_same_name(
        self, adapter: ArtifactStorePort
    ) -> None:
        a = adapter.write_predictions(_predictions(), "test")
        b = adapter.write_predictions(_predictions(), "test")
        assert a == b

    def test_write_model_card_idempotent(
        self, adapter: ArtifactStorePort
    ) -> None:
        a = adapter.write_model_card(_model_card())
        b = adapter.write_model_card(_model_card())
        assert a == b


class TestInMemoryFakeArtifactStoreContract(ArtifactStorePortContract):
    @pytest.fixture
    def adapter(self) -> ArtifactStorePort:
        return InMemoryFakeArtifactStore()


class TestInMemoryFakeArtifactStoreInspection:
    """Inspection surface specific to the in-memory fake."""

    def test_written_card_is_retrievable(self) -> None:
        store = InMemoryFakeArtifactStore()
        card = _model_card()
        path = store.write_model_card(card)
        assert store.get(path) is card

    def test_written_predictions_retrievable(self) -> None:
        store = InMemoryFakeArtifactStore()
        preds = _predictions()
        path = store.write_predictions(preds, "test")
        assert store.get(path) is preds

    def test_written_thresholds_retrievable(self) -> None:
        store = InMemoryFakeArtifactStore()
        ts = _thresholds()
        path = store.write_thresholds(ts)
        assert store.get(path) is ts

    def test_written_metric_report_retrievable(self) -> None:
        store = InMemoryFakeArtifactStore()
        report = _metric_report()
        path = store.write_metric_report(report)
        assert store.get(path) is report

    def test_get_unknown_path_raises_key_error(self) -> None:
        store = InMemoryFakeArtifactStore()
        with pytest.raises(KeyError):
            store.get("memory://unknown")


class TestFilesystemArtifactStoreContract(ArtifactStorePortContract):
    @pytest.fixture
    def adapter(self, tmp_path: Path) -> ArtifactStorePort:
        return FilesystemArtifactStore(tmp_path)
