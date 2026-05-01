"""Unit tests for ``FilesystemArtifactStore``.

These tests exercise the on-disk publication-ready artifact layout. Each test
uses pytest's ``tmp_path`` fixture for isolation. Round-trip tests verify
that JSON serialization preserves all dataclass fields and that the CSV
predictions file is parseable with stdlib ``csv``.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from harness.adapters.fs.artifact_store import FilesystemArtifactStore
from harness.domain.types import (
    MetricInterval,
    MetricReport,
    ModelCard,
    PerClassMetric,
    Predictions,
    ThresholdSet,
)

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _interval(point: float = 0.7, lower: float = 0.6, upper: float = 0.8) -> MetricInterval:
    return MetricInterval(point=point, lower=lower, upper=upper)


def _metric_report() -> MetricReport:
    return MetricReport(
        macro_f1=_interval(0.55, 0.50, 0.60),
        macro_auroc=_interval(0.82, 0.79, 0.85),
        macro_auprc=_interval(0.40, 0.35, 0.45),
        per_class=(
            PerClassMetric(
                label="a",
                f1=_interval(0.60, 0.55, 0.65),
                auroc=_interval(0.81, 0.78, 0.84),
                auprc=_interval(0.42, 0.37, 0.47),
                support=12,
            ),
            PerClassMetric(
                label="b",
                f1=_interval(0.50, 0.45, 0.55),
                auroc=_interval(0.83, 0.80, 0.86),
                auprc=_interval(0.38, 0.33, 0.43),
                support=9,
            ),
        ),
        n_bootstrap=64,
        seed=7,
    )


def _model_card() -> ModelCard:
    return ModelCard(
        name="cxr-demo",
        version="v1.2.3",
        created_at=datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC),
        backbone="resnet50",
        head="linear",
        calibrator="platt",
        threshold_method="pr_sweep",
        label_names=("a", "b"),
        train_size=100,
        val_size=20,
        test_size=30,
        config_hash="deadbeef",
        metrics=_metric_report(),
        notes="unit-test card",
    )


def _thresholds() -> ThresholdSet:
    return ThresholdSet(
        label_names=("a", "b"),
        thresholds=(0.4321, 0.5678),
        method="pr_sweep",
        shrinkage=0.1,
        clamp_lo=0.05,
        clamp_hi=0.95,
    )


def _predictions() -> Predictions:
    return Predictions(
        sample_ids=("s0", "s1", "s2"),
        label_names=("a", "b"),
        values=np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_creates_root_dir_if_missing(tmp_path: Path) -> None:
    root = tmp_path / "does" / "not" / "exist"
    store = FilesystemArtifactStore(root)
    path = store.write_thresholds(_thresholds())
    assert root.exists() and root.is_dir()
    assert Path(path).exists()


def test_write_methods_return_existing_paths(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    paths = [
        store.write_model_card(_model_card()),
        store.write_thresholds(_thresholds()),
        store.write_metric_report(_metric_report()),
        store.write_predictions(_predictions(), name="test"),
    ]
    for p in paths:
        assert isinstance(p, str)
        assert Path(p).exists()


def test_write_methods_return_absolute_paths(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    p = store.write_model_card(_model_card())
    assert Path(p).is_absolute()


def test_predictions_filename_includes_name_arg(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    p = store.write_predictions(_predictions(), name="val")
    path = Path(p)
    assert path.name == "val.csv"
    assert path.parent.name == "predictions"


def test_round_trip_model_card(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    card = _model_card()
    path = store.write_model_card(card)

    with Path(path).open(encoding="utf-8") as fp:
        data = json.load(fp)

    assert data["name"] == card.name
    assert data["version"] == card.version
    assert data["backbone"] == card.backbone
    assert data["head"] == card.head
    assert data["calibrator"] == card.calibrator
    assert data["threshold_method"] == card.threshold_method
    assert tuple(data["label_names"]) == card.label_names
    assert data["train_size"] == card.train_size
    assert data["val_size"] == card.val_size
    assert data["test_size"] == card.test_size
    assert data["config_hash"] == card.config_hash
    assert data["notes"] == card.notes
    # created_at serialized as ISO 8601 string
    assert datetime.fromisoformat(data["created_at"]) == card.created_at
    # metrics nested
    assert data["metrics"]["n_bootstrap"] == card.metrics.n_bootstrap
    assert data["metrics"]["seed"] == card.metrics.seed
    assert data["metrics"]["macro_f1"]["point"] == card.metrics.macro_f1.point


def test_round_trip_thresholds(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    ts = _thresholds()
    path = store.write_thresholds(ts)

    with Path(path).open(encoding="utf-8") as fp:
        data = json.load(fp)

    assert "per_label" in data
    assert data["per_label"]["a"] == ts.thresholds[0]
    assert data["per_label"]["b"] == ts.thresholds[1]
    assert data["method"] == ts.method
    assert data["shrinkage"] == ts.shrinkage
    assert data["clamp_lo"] == ts.clamp_lo
    assert data["clamp_hi"] == ts.clamp_hi


def test_round_trip_metric_report(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    report = _metric_report()
    path = store.write_metric_report(report)

    with Path(path).open(encoding="utf-8") as fp:
        data = json.load(fp)

    assert data["n_bootstrap"] == report.n_bootstrap
    assert data["seed"] == report.seed
    for macro_key, macro in (
        ("macro_f1", report.macro_f1),
        ("macro_auroc", report.macro_auroc),
        ("macro_auprc", report.macro_auprc),
    ):
        assert data[macro_key]["point"] == macro.point
        assert data[macro_key]["low"] == macro.lower
        assert data[macro_key]["high"] == macro.upper
    # per-class
    assert len(data["per_class"]) == len(report.per_class)
    for got, expected in zip(data["per_class"], report.per_class, strict=True):
        assert got["label"] == expected.label
        assert got["support"] == expected.support
        assert got["f1"]["point"] == expected.f1.point
        assert got["f1"]["low"] == expected.f1.lower
        assert got["f1"]["high"] == expected.f1.upper
        assert got["auroc"]["point"] == expected.auroc.point
        assert got["auprc"]["point"] == expected.auprc.point


def test_predictions_csv_format(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    preds = _predictions()
    path = store.write_predictions(preds, name="test")

    with Path(path).open(encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        rows = list(reader)

    # header includes a sample id column followed by label names
    assert len(rows) == len(preds.sample_ids) + 1
    header = rows[0]
    assert header[1:] == list(preds.label_names)
    # data rows
    for i, sid in enumerate(preds.sample_ids):
        data_row = rows[1 + i]
        assert data_row[0] == sid
        values = [int(v) for v in data_row[1:]]
        assert values == preds.values[i].tolist()


def test_predictions_loadable_with_numpy(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    preds = _predictions()
    path = store.write_predictions(preds, name="test")
    arr = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=range(1, 1 + len(preds.label_names)),
        dtype=np.int8,
    )
    assert arr.shape == preds.values.shape
    np.testing.assert_array_equal(arr, preds.values)


def test_idempotent_overwrite_thresholds(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    p1 = store.write_thresholds(_thresholds())
    p2 = store.write_thresholds(_thresholds())
    assert p1 == p2
    assert Path(p1).exists()


def test_idempotent_overwrite_model_card(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    p1 = store.write_model_card(_model_card())
    p2 = store.write_model_card(_model_card())
    assert p1 == p2


def test_idempotent_overwrite_metric_report(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    p1 = store.write_metric_report(_metric_report())
    p2 = store.write_metric_report(_metric_report())
    assert p1 == p2


def test_idempotent_overwrite_predictions(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    p1 = store.write_predictions(_predictions(), name="test")
    p2 = store.write_predictions(_predictions(), name="test")
    assert p1 == p2


def test_layout_locations(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    card_path = store.write_model_card(_model_card())
    thr_path = store.write_thresholds(_thresholds())
    rep_path = store.write_metric_report(_metric_report())
    pred_path = store.write_predictions(_predictions(), name="test")
    assert Path(card_path).name == "model_card.json"
    assert Path(thr_path).name == "thresholds.json"
    assert Path(rep_path).name == "metric_report.json"
    assert Path(pred_path).name == "test.csv"
    assert Path(pred_path).parent.name == "predictions"


def test_overwrite_replaces_content(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    first = ThresholdSet(
        label_names=("a",),
        thresholds=(0.1,),
        method="manual",
        shrinkage=0.0,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )
    second = ThresholdSet(
        label_names=("a",),
        thresholds=(0.9,),
        method="manual",
        shrinkage=0.0,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )
    store.write_thresholds(first)
    path = store.write_thresholds(second)
    with Path(path).open(encoding="utf-8") as fp:
        data = json.load(fp)
    assert data["per_label"]["a"] == 0.9


def test_predictions_ndarray_serialized_as_int(tmp_path: Path) -> None:
    """CSV must contain plain integers, not numpy repr."""
    store = FilesystemArtifactStore(tmp_path)
    store.write_predictions(_predictions(), name="test")
    text = (tmp_path / "predictions" / "test.csv").read_text(encoding="utf-8")
    assert "np.int8" not in text
    assert "[" not in text


@pytest.mark.parametrize("name", ["val", "test", "holdout-2025"])
def test_predictions_name_variants(tmp_path: Path, name: str) -> None:
    store = FilesystemArtifactStore(tmp_path)
    p = store.write_predictions(_predictions(), name=name)
    assert Path(p).name == f"{name}.csv"
