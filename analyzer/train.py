"""Train persisted per-finding calibration profiles over extracted CXR metrics.

Usage:
    python -m analyzer.train --labels Data_Entry_2017_v2020.csv --report results/report.json \
        --out-profile models/cxr_profile.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from analyzer.evaluate import Confusion, is_positive, load_labels
from analyzer.features import METRIC_KEYS, METRIC_SCHEMA_VERSION
from analyzer.m4_findings import FINDINGS
from analyzer.predict import build_report_record
from analyzer.profile import PROFILE_FORMAT_VERSION, PROFILE_TYPE, predict_with_profile, save_profile

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Findings where the val-F1 optimizer tends to push thresholds too high (0.80+)
# by sacrificing all recall to avoid FPs on a rare, clinically vague finding.
# These ceilings accept more FPs in exchange for not missing every positive.
THRESHOLD_CEILINGS: dict[str, float] = {
    "consolidation": 0.58,
    "atelectasis": 0.50,
}

FEATURE_SUBSETS: dict[str, tuple[str, ...]] = dict.fromkeys(FINDINGS, METRIC_KEYS)


@dataclass(frozen=True)
class SplitRecords:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]


def load_report(path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(path.read_text())
    if not data:
        raise ValueError(f"{path} did not contain any report records")
    return data


def deterministic_split(
    records: Sequence[dict[str, Any]],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> SplitRecords:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must leave a non-empty test fraction")

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for rec in records:
        bucket = int.from_bytes(hashlib.sha1(rec["image"].encode("utf-8")).digest()[:8], "big") / 2**64
        if bucket < train_frac:
            train.append(rec)
        elif bucket < train_frac + val_frac:
            val.append(rec)
        else:
            test.append(rec)
    return SplitRecords(train=train, val=val, test=test)


def _probability_threshold(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    if not probabilities:
        return 0.5
    candidates = sorted(set(probabilities))
    best_f1 = -1.0
    best_threshold = 0.5
    for i, value in enumerate(candidates):
        threshold = (value + candidates[i + 1]) / 2 if i + 1 < len(candidates) else value
        preds = [p >= threshold for p in probabilities]
        cm = confusion_from_predictions(preds, labels)
        if cm.f1 > best_f1:
            best_f1 = cm.f1
            best_threshold = threshold
    return float(best_threshold)


def confusion_from_predictions(predictions: Iterable[bool], labels: Iterable[bool]) -> Confusion:
    cm = Confusion()
    for pred, actual in zip(predictions, labels, strict=True):
        if pred and actual:
            cm.tp += 1
        elif pred and not actual:
            cm.fp += 1
        elif not pred and actual:
            cm.fn += 1
        else:
            cm.tn += 1
    return cm


def _metrics_matrix(records: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.array([[float(rec["metrics"][name]) for name in feature_names] for rec in records], dtype=float)


def _label_vector(
    records: Sequence[dict[str, Any]],
    labels: dict[str, set[str]],
    finding_key: str,
) -> np.ndarray:
    return np.array([is_positive(labels[rec["image"]], finding_key) for rec in records], dtype=bool)


def _fit_one_finding(
    finding_key: str,
    split_records: SplitRecords,
    labels: dict[str, set[str]],
) -> tuple[dict[str, Any], Any, dict[str, list[float]]]:
    feature_names = list(FEATURE_SUBSETS[finding_key])
    X_train = _metrics_matrix(split_records.train, feature_names)
    y_train = _label_vector(split_records.train, labels, finding_key)
    X_val = _metrics_matrix(split_records.val, feature_names)
    y_val = _label_vector(split_records.val, labels, finding_key)
    X_test = _metrics_matrix(split_records.test, feature_names)
    y_test = _label_vector(split_records.test, labels, finding_key)

    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{finding_key}: train split has only one class")
    if len(np.unique(y_val)) < 2:
        raise ValueError(f"{finding_key}: validation split has only one class")
    if len(np.unique(y_test)) < 2:
        raise ValueError(f"{finding_key}: test split has only one class")

    base = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=0,
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train.astype(int))

    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    threshold = _probability_threshold(val_probs.tolist(), y_val.tolist())
    ceiling = THRESHOLD_CEILINGS.get(finding_key)
    threshold_source = "validation_f1"
    if ceiling is not None and threshold > ceiling:
        threshold = ceiling
        threshold_source = "ceiling"

    def summarize(probs: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        preds = probs >= threshold
        cm = confusion_from_predictions(preds.tolist(), y_true.tolist())
        n = len(y_true)
        return {
            "threshold": float(threshold),
            "precision": cm.precision,
            "recall": cm.recall,
            "f1": cm.f1,
            "tp": cm.tp,
            "fp": cm.fp,
            "fn": cm.fn,
            "tn": cm.tn,
            "prevalence": float(y_true.mean()) if n else 0.0,
            "fired_rate": float(preds.mean()) if n else 0.0,
        }

    perm = permutation_importance(model, X_val, y_val.astype(int), n_repeats=5, random_state=0)
    model_payload = {
        "feature_names": feature_names,
        "feature_importances": [float(x) for x in perm.importances_mean],
        "model_class": "CalibratedClassifierCV(HistGradientBoostingClassifier)",
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "train_metrics": summarize(train_probs, y_train),
        "validation_metrics": summarize(val_probs, y_val),
        "test_metrics": summarize(test_probs, y_test),
    }
    split_probabilities = {
        "train": [float(x) for x in train_probs],
        "val": [float(x) for x in val_probs],
        "test": [float(x) for x in test_probs],
    }
    return model_payload, model, split_probabilities


def _assemble_profile(
    split_records: SplitRecords,
    labels: dict[str, set[str]],
    report_path: Path,
    labels_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, list[float]]], dict[str, Any]]:
    findings: dict[str, Any] = {}
    split_probabilities: dict[str, dict[str, list[float]]] = {}
    models_dict: dict[str, Any] = {}
    for finding_key in FINDINGS:
        payload, fitted_model, probs = _fit_one_finding(finding_key, split_records, labels)
        findings[finding_key] = payload
        models_dict[finding_key] = fitted_model
        split_probabilities[finding_key] = probs

    profile = {
        "profile_type": PROFILE_TYPE,
        "format_version": PROFILE_FORMAT_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "training": {
            "report_path": str(report_path),
            "labels_path": str(labels_path),
            "split_policy": {
                "kind": "deterministic_sha1",
                "train_frac": TRAIN_FRAC,
                "val_frac": VAL_FRAC,
                "test_frac": TEST_FRAC,
            },
            "dataset_size": len(split_records.train) + len(split_records.val) + len(split_records.test),
            "split_sizes": {
                "train": len(split_records.train),
                "val": len(split_records.val),
                "test": len(split_records.test),
            },
        },
        "findings": findings,
    }
    return profile, split_probabilities, models_dict


def _predict_records(
    records: Sequence[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        predictions = predict_with_profile(rec["metrics"], profile)
        out.append(build_report_record(rec["image"], rec["metrics"], predictions))
    return out


def train_profile(
    report_path: Path,
    labels_path: Path,
    out_profile: Path,
) -> dict[str, Any]:
    labels = load_labels(labels_path)
    records = [r for r in load_report(report_path) if r["image"] in labels]
    if not records:
        raise SystemExit("No report records overlap with the labels CSV — check filenames")

    split_records = deterministic_split(records)
    profile, _, models_dict = _assemble_profile(split_records, labels, report_path, labels_path)
    # Inject models for _predict_records, then save both JSON + joblib
    profile["_models"] = models_dict
    save_profile(profile, out_profile, models=models_dict)

    base = out_profile.with_suffix("")
    for split_name, split_data in (
        ("train", split_records.train),
        ("val", split_records.val),
        ("test", split_records.test),
    ):
        split_path = base.with_name(f"{base.name}.{split_name}_report.json")
        split_path.write_text(json.dumps(_predict_records(split_data, profile), indent=2))

    return profile


def _print_summary(profile: dict[str, Any]) -> None:
    training = profile["training"]
    sizes = training["split_sizes"]
    print(
        "  Trained calibration profile — "
        f"{training['dataset_size']} labeled records "
        f"({sizes['train']} train / {sizes['val']} val / {sizes['test']} test)"
    )
    for key, model in profile["findings"].items():
        test_metrics = model["test_metrics"]
        print(
            f"  {key:<18} threshold={model['threshold']:.4f} "
            f"P={test_metrics['precision']:.2f} "
            f"R={test_metrics['recall']:.2f} "
            f"F1={test_metrics['f1']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="NIH Data_Entry_*.csv")
    parser.add_argument("--report", type=Path, default=Path("results/report.json"))
    parser.add_argument("--out-profile", required=True, type=Path, help="where to write the profile JSON")
    args = parser.parse_args()

    profile = train_profile(args.report, args.labels, args.out_profile)
    _print_summary(profile)
    print(f"\n  Profile saved → {args.out_profile}")
    base = args.out_profile.with_suffix("")
    print(f"  Train report   → {base.with_name(f'{base.name}.train_report.json')}")
    print(f"  Val report     → {base.with_name(f'{base.name}.val_report.json')}")
    print(f"  Test report    → {base.with_name(f'{base.name}.test_report.json')}")


if __name__ == "__main__":
    main()
