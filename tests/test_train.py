from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from analyzer.features import METRIC_KEYS
from analyzer.m4_findings import FINDINGS
from analyzer.predict import build_report_record
from analyzer.profile import load_profile
from analyzer.train import _probability_threshold, deterministic_split, train_profile


def _blank_predictions() -> dict[str, dict[str, bool | None]]:
    return {
        key: {"detected": False, "probability": None}
        for key in FINDINGS
    }


def _synthetic_metrics(i: int) -> tuple[dict[str, float], list[str]]:
    metrics = dict.fromkeys(METRIC_KEYS, 0.1)
    labels: list[str] = []

    if i % 7 in {0, 1}:
        metrics["ctr"] = 0.95
        labels.append("Cardiomegaly")
    else:
        metrics["ctr"] = 0.05

    if i % 9 == 0:
        metrics["ptx_left_mean"] = 0.05
        metrics["ptx_left_std"] = 0.05
        metrics["ptx_right_mean"] = 0.05
        metrics["ptx_right_std"] = 0.05
        labels.append("Pneumothorax")
    else:
        metrics["ptx_left_mean"] = 0.9
        metrics["ptx_left_std"] = 0.9
        metrics["ptx_right_mean"] = 0.9
        metrics["ptx_right_std"] = 0.9

    if i % 5 == 0:
        metrics["basal_opacity"] = 0.95
        labels.append("Effusion")
    else:
        metrics["basal_opacity"] = 0.05

    if i % 11 == 0:
        metrics["bilateral_haze"] = 0.95
        labels.append("Edema")
    else:
        metrics["bilateral_haze"] = 0.05

    if i % 13 == 0:
        metrics["bilateral_haze"] = max(metrics["bilateral_haze"], 0.90)
        metrics["basal_opacity"] = max(metrics["basal_opacity"], 0.90)
        metrics["focal_variance"] = 0.01
        labels.append("Consolidation")
    elif "Edema" not in labels:
        metrics["focal_variance"] = 0.04

    if i % 6 == 0:
        metrics["horiz_band"] = 0.95
        labels.append("Atelectasis")
    else:
        metrics["horiz_band"] = 0.05

    if i % 10 == 0:
        metrics["diaphragm_pos"] = 0.95
        labels.append("Emphysema")
    else:
        metrics["diaphragm_pos"] = 0.05

    if i % 4 == 0:
        metrics["focal_variance"] = 0.95
        labels.append("Mass")
    elif "Consolidation" not in labels:
        metrics["focal_variance"] = 0.02

    return metrics, labels or ["No Finding"]


def _write_synthetic_inputs(tmp_path: Path, n: int = 240) -> tuple[Path, Path]:
    report_path = tmp_path / "report.json"
    labels_path = tmp_path / "labels.csv"

    records = []
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Index", "Finding Labels"])
        for i in range(n):
            image = f"img_{i:04d}.png"
            metrics, labels = _synthetic_metrics(i)
            writer.writerow([image, "|".join(labels)])
            records.append(build_report_record(image, metrics, _blank_predictions()))
    report_path.write_text(json.dumps(records, indent=2))
    return report_path, labels_path


def test_deterministic_split_is_stable_and_disjoint() -> None:
    records = [{"image": f"img_{i:04d}.png", "metrics": {}} for i in range(40)]
    split1 = deterministic_split(records)
    split2 = deterministic_split(records)
    names1 = {rec["image"] for rec in split1.train + split1.val + split1.test}
    names2 = {rec["image"] for rec in split2.train + split2.val + split2.test}
    assert names1 == names2 == {rec["image"] for rec in records}
    assert {rec["image"] for rec in split1.train}.isdisjoint({rec["image"] for rec in split1.val})
    assert {rec["image"] for rec in split1.train}.isdisjoint({rec["image"] for rec in split1.test})
    assert {rec["image"] for rec in split1.val}.isdisjoint({rec["image"] for rec in split1.test})


def test_probability_threshold_helper_maximizes_f1() -> None:
    threshold = _probability_threshold([0.10, 0.20, 0.80, 0.90], [False, False, True, True])
    assert threshold == pytest.approx(0.50, abs=1e-9)


@pytest.mark.slow
def test_train_profile_writes_profile_and_probability_reports(tmp_path: Path) -> None:
    report_path, labels_path = _write_synthetic_inputs(tmp_path)
    out_profile = tmp_path / "models" / "cxr_profile.json"
    profile = train_profile(report_path, labels_path, out_profile)

    assert out_profile.exists()
    assert out_profile.with_suffix(".joblib").exists()
    loaded = load_profile(out_profile)
    assert loaded["metric_schema_version"]
    assert loaded["findings"]["cardiomegaly"]["threshold_source"] == "validation_f1"
    assert "feature_importances" in loaded["findings"]["cardiomegaly"]

    base = out_profile.with_suffix("")
    test_report = base.with_name(f"{base.name}.test_report.json")
    assert test_report.exists()
    records = json.loads(test_report.read_text())
    assert records
    records_with_ddx = [r for r in records if r["ddx"]]
    assert records_with_ddx, "Expected at least one record with detected findings in test report"
    entry = records_with_ddx[0]["ddx"][0]
    assert "probability" in entry
    assert "considerations" in entry
    assert "confidence" in entry
    assert set(profile["findings"]) == set(FINDINGS)
