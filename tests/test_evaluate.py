"""Unit tests for analyzer/evaluate.py.

Uses small synthetic records + a synthetic NIH-style labels CSV so we can
hand-verify confusion-matrix counts and Youden's J selection.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from analyzer.evaluate import (
    Confusion,
    confusion_matrix,
    is_positive,
    load_labels,
    youden_threshold,
)
from analyzer.main import THRESHOLDS


def _record(image: str, detected: dict[str, bool], metrics: dict[str, float]) -> dict[str, Any]:
    ddx = [
        {"finding": key, "name": key, "tier": 1, "tier_label": "", "probability": None,
         "confidence": None, "description": "", "m4_action": "", "considerations": []}
        for key in THRESHOLDS
        if detected.get(key, False)
    ]
    return {"image": image, "ddx": ddx, "metrics": metrics}


_METRIC_KEYS = (
    "ctr",
    "ptx_left_mean",
    "ptx_right_mean",
    "ptx_left_std",
    "ptx_right_std",
    "ptx_left_edge_density",
    "ptx_right_edge_density",
    "basal_opacity",
    "bilateral_haze",
    "diaphragm_pos",
    "diaphragm_flatness",
    "focal_variance",
    "horiz_band",
    "lower_horiz_band",
)


def _write_labels(path: Path, rows: dict[str, str]) -> None:
    """rows: image filename → pipe-separated NIH label string."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image Index", "Finding Labels"])
        for img, lbl in rows.items():
            w.writerow([img, lbl])


def test_load_labels_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "labels.csv"
    _write_labels(
        p,
        {
            "a.png": "Cardiomegaly",
            "b.png": "Effusion|Atelectasis",
            "c.png": "No Finding",
        },
    )
    out = load_labels(p)
    assert out["a.png"] == {"Cardiomegaly"}
    assert out["b.png"] == {"Effusion", "Atelectasis"}
    assert out["c.png"] == {"No Finding"}


def test_is_positive_focal_opacity_union() -> None:
    assert is_positive({"Mass"}, "focal_opacity")
    assert is_positive({"Nodule"}, "focal_opacity")
    # Infiltration is a diffuse airspace pattern, NOT a focal lesion;
    # it must NOT count as positive for focal_opacity.
    assert not is_positive({"Infiltration"}, "focal_opacity")
    assert not is_positive({"Effusion"}, "focal_opacity")


def test_is_positive_consolidation_includes_pneumonia() -> None:
    """Pneumonia in NIH is radiographically a consolidation; both must count."""
    assert is_positive({"Consolidation"}, "consolidation")
    assert is_positive({"Pneumonia"}, "consolidation")


def test_is_positive_simple_mapping() -> None:
    assert is_positive({"Cardiomegaly"}, "cardiomegaly")
    assert not is_positive({"Cardiomegaly"}, "pleural_effusion")
    assert is_positive({"Effusion"}, "pleural_effusion")


def test_confusion_matrix_counts() -> None:
    metrics = dict.fromkeys(_METRIC_KEYS, 0.0)
    records = [
        # TP for cardiomegaly
        _record("tp.png", detected={"cardiomegaly": True}, metrics=metrics),
        # FP for cardiomegaly (predicted, not actual)
        _record("fp.png", detected={"cardiomegaly": True}, metrics=metrics),
        # FN for cardiomegaly (not predicted, but actual)
        _record("fn.png", detected={"cardiomegaly": False}, metrics=metrics),
        # TN for cardiomegaly
        _record("tn.png", detected={"cardiomegaly": False}, metrics=metrics),
    ]
    labels = {
        "tp.png": {"Cardiomegaly"},
        "fp.png": {"No Finding"},
        "fn.png": {"Cardiomegaly"},
        "tn.png": {"No Finding"},
    }
    cm = confusion_matrix(records, labels)
    c = cm["cardiomegaly"]
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.precision == 0.5
    assert c.recall == 0.5
    assert c.f1 == 0.5


def test_confusion_matrix_skips_unlabeled() -> None:
    metrics = dict.fromkeys(_METRIC_KEYS, 0.0)
    records = [_record("orphan.png", detected={"cardiomegaly": True}, metrics=metrics)]
    cm = confusion_matrix(records, labels={})
    assert cm["cardiomegaly"] == Confusion(0, 0, 0, 0)


def test_youden_picks_separable_split() -> None:
    """When positives have ctr>=0.6 and negatives have ctr<=0.4, the optimal
    threshold sits in between and J should be 1.0."""
    base_metrics = dict.fromkeys(_METRIC_KEYS, 0.0)
    records: list[dict[str, Any]] = []
    labels: dict[str, set[str]] = {}
    for i, ctr in enumerate([0.10, 0.20, 0.30, 0.40]):
        img = f"neg{i}.png"
        m = dict(base_metrics, ctr=ctr)
        records.append(_record(img, {}, m))
        labels[img] = {"No Finding"}
    for i, ctr in enumerate([0.60, 0.70, 0.80, 0.90]):
        img = f"pos{i}.png"
        m = dict(base_metrics, ctr=ctr)
        records.append(_record(img, {}, m))
        labels[img] = {"Cardiomegaly"}

    out = youden_threshold(records, labels, "cardiomegaly", "ctr")
    assert out is not None
    j, t = out
    assert j == pytest.approx(1.0, abs=1e-9)
    # Conventional midpoint of (highest negative, lowest positive) = 0.50.
    assert t == pytest.approx(0.50, abs=1e-9)


def test_youden_handles_ties_across_classes() -> None:
    """When the same metric value appears in both classes, the algorithm must
    process the tie atomically — evaluating J mid-tie would over-credit a
    threshold that misclassifies tied positives.

    On this dataset, two distinct thresholds both yield J = 2/3:
      t=0.35: keeps the tied 0.50-pos (TP) and tied 0.50-neg (FP) → 3/3 − 1/3
      t=0.60: drops both tied 0.50 samples (1 FN, 0 FP) → 2/3 − 0/3
    The implementation uses strict `>` comparison on J, so it keeps the FIRST
    candidate it sees and reports t=0.35. Whichever it returns, J must equal
    2/3 — so we assert J directly and only sanity-check that t is one of the
    two valid answers.
    """
    base_metrics = dict.fromkeys(_METRIC_KEYS, 0.0)
    records: list[dict[str, Any]] = []
    labels: dict[str, set[str]] = {}
    pairs = [
        (0.10, False),
        (0.20, False),
        (0.50, False),
        (0.50, True),
        (0.70, True),
        (0.90, True),
    ]
    for i, (ctr, pos) in enumerate(pairs):
        img = f"x{i}.png"
        records.append(_record(img, {}, dict(base_metrics, ctr=ctr)))
        labels[img] = {"Cardiomegaly"} if pos else {"No Finding"}
    out = youden_threshold(records, labels, "cardiomegaly", "ctr")
    assert out is not None
    j, t = out
    assert j == pytest.approx(2 / 3, abs=1e-9)
    assert t == pytest.approx(0.35, abs=1e-9) or t == pytest.approx(0.60, abs=1e-9)
    # Whichever threshold is returned, applying `metric > t` to the ties must
    # produce a confusion matrix consistent with J = 2/3.
    pred_pos = sum(1 for v, _ in [(p[0], p[1]) for p in pairs] if v > t and _)
    pred_neg_actual_pos = sum(1 for v, p in pairs if not (v > t) and p)
    pred_pos_actual_neg = sum(1 for v, p in pairs if v > t and not p)
    tpr = pred_pos / 3
    fpr = pred_pos_actual_neg / 3
    assert tpr - fpr == pytest.approx(2 / 3, abs=1e-9)
    assert pred_neg_actual_pos + pred_pos == 3


def test_youden_returns_none_when_one_class_empty() -> None:
    records = [
        _record("a.png", {}, dict.fromkeys(_METRIC_KEYS, 0.0) | {"ctr": 0.5})
    ]
    labels = {"a.png": {"No Finding"}}
    assert youden_threshold(records, labels, "cardiomegaly", "ctr") is None
