"""Evaluate threshold- or profile-based detections against NIH ChestX-ray14 labels.

Usage:
    python -m analyzer.evaluate --labels Data_Entry_2017_v2020.csv
    python -m analyzer.evaluate --labels Data_Entry_2017_v2020.csv --suggest

The labels CSV is the file shipped with the NIH ChestX-ray14 release. Each row
has an `Image Index` (filename) and `Finding Labels` (pipe-separated NIH
labels). Only images present in BOTH the report.json and the labels file are
scored. The report may come from the legacy threshold fallback or from a saved
calibration profile; evaluation only depends on `findings.<key>.detected`.

Threshold suggestions use Youden's J (TPR - FPR) on the primary metric for
each single-metric finding. Compound findings (pneumothorax, consolidation)
are reported but not auto-suggested — they need joint optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyzer.m4_findings import FINDINGS

# Mapping from our heuristic key to NIH ChestX-ray14 labels that count as positive.
# A few mappings are deliberately set-valued: NIH labels Consolidation and Pneumonia
# both correspond to the radiographic finding our `consolidation` rule flags;
# focal_opacity is the union of focal lesions (Mass, Nodule).
# `Infiltration` is intentionally NOT in focal_opacity — it is a diffuse airspace
# pattern, which our `consolidation` rule already covers via bilateral haze.
NIH_POSITIVE_LABELS: dict[str, frozenset[str]] = {
    "cardiomegaly": frozenset({"Cardiomegaly"}),
    "pneumothorax": frozenset({"Pneumothorax"}),
    "pleural_effusion": frozenset({"Effusion"}),
    "pulmonary_edema": frozenset({"Edema"}),
    "consolidation": frozenset({"Consolidation", "Pneumonia"}),
    "atelectasis": frozenset({"Atelectasis"}),
    "emphysema": frozenset({"Emphysema"}),
    "focal_opacity": frozenset({"Mass", "Nodule"}),
}

# For single-metric findings, the primary feature each rule keys off of.
# Compound rules (pneumothorax, consolidation) are intentionally absent.
PRIMARY_METRIC = {
    "cardiomegaly": "ctr",
    "pleural_effusion": "basal_opacity",
    "pulmonary_edema": "bilateral_haze",
    "atelectasis": "horiz_band",
    "emphysema": "diaphragm_pos",
    "focal_opacity": "focal_variance",
}


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def load_labels(csv_path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if fields is None or "Image Index" not in fields or "Finding Labels" not in fields:
            raise ValueError(
                f"{csv_path} is missing required columns 'Image Index' and 'Finding Labels'; got {fields!r}"
            )
        for row in reader:
            out[row["Image Index"]] = set(row["Finding Labels"].split("|"))
    return out


def is_positive(nih_labels: set[str], our_key: str) -> bool:
    targets = NIH_POSITIVE_LABELS.get(our_key)
    return bool(targets and (targets & nih_labels))


def confusion_matrix(records: Iterable[dict[str, Any]], labels: dict[str, set[str]]) -> dict[str, Confusion]:
    """Build per-finding confusion counts.

    Note: per-finding rows are NOT mutually exclusive — a single NIH image can
    carry multiple labels (e.g. `Effusion|Mass`), so it can be a positive case
    for both `pleural_effusion` and `focal_opacity` in the same row. Don't sum
    across findings.
    """
    out = {key: Confusion() for key in FINDINGS}
    for r in records:
        nih = labels.get(r["image"])
        if nih is None:
            continue
        for key in FINDINGS:
            pred = r["findings"][key]["detected"]
            actual = is_positive(nih, key)
            cm = out[key]
            if pred and actual:
                cm.tp += 1
            elif pred and not actual:
                cm.fp += 1
            elif not pred and actual:
                cm.fn += 1
            else:
                cm.tn += 1
    return out


def youden_threshold(
    records: Iterable[dict[str, Any]],
    labels: dict[str, set[str]],
    our_key: str,
    metric_key: str,
) -> tuple[float, float] | None:
    """Find the metric threshold that maximizes TPR - FPR.

    Predicate is `metric > threshold` (matching `main.THRESHOLDS` for
    single-metric findings). Returns `(J, threshold)` or `None` when one class
    is empty.

    Tied metric values across classes are processed atomically — J is only
    evaluated *between* distinct values. The reported threshold is the midpoint
    between the optimal flipped value and the next distinct value, which is the
    standard ROC-threshold convention (robust to floating-point ties).
    """
    samples: list[tuple[float, bool]] = []
    for r in records:
        nih = labels.get(r["image"])
        if nih is None:
            continue
        samples.append((r["metrics"][metric_key], is_positive(nih, our_key)))

    P = sum(1 for _, a in samples if a)
    N = len(samples) - P
    if P == 0 or N == 0:
        return None

    samples.sort(key=lambda s: s[0])

    # Start with threshold below the smallest value: every sample predicted positive.
    tp, fn = P, 0
    fp, tn = N, 0
    best_j: float = float("-inf")
    best_t: float = samples[-1][0]

    n = len(samples)
    i = 0
    while i < n:
        # Drain every sample tied at samples[i][0] before evaluating J.
        tied_value = samples[i][0]
        j_idx = i
        while j_idx < n and samples[j_idx][0] == tied_value:
            if samples[j_idx][1]:
                tp -= 1
                fn += 1
            else:
                fp -= 1
                tn += 1
            j_idx += 1

        j_score = (tp / P) - (fp / N)
        if j_score > best_j:
            best_j = j_score
            # Threshold sits between this distinct value and the next; if no
            # next value exists, fall back to the value itself.
            best_t = (tied_value + samples[j_idx][0]) / 2 if j_idx < n else tied_value
        i = j_idx

    return best_j, best_t


def print_confusion(cm: dict[str, Confusion], n: int) -> None:
    print(f"\n  Evaluated {n} labeled records")
    header = (
        f"  {'finding':<20} {'prev':>5}  {'fired':>5}    "
        f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>5}    "
        f"{'prec':>4} {'rec':>4} {'F1':>4}"
    )
    print(header)
    print("  " + "─" * 78)
    for key, m in cm.items():
        prev = (m.tp + m.fn) / n if n else 0.0
        fired = (m.tp + m.fp) / n if n else 0.0
        print(
            f"  {key:<20} {prev * 100:4.1f}%  {fired * 100:4.1f}%    "
            f"{m.tp:>4} {m.fp:>4} {m.fn:>4} {m.tn:>5}    "
            f"{m.precision:.2f} {m.recall:.2f} {m.f1:.2f}"
        )


def print_suggestions(records: list[dict[str, Any]], labels: dict[str, set[str]]) -> None:
    print("\n  Threshold suggestions via Youden's J (single-metric findings only):")
    print(f"  {'finding':<20} {'metric':<18} {'threshold':>11}  {'J':>5}")
    print("  " + "─" * 60)
    for our_key, metric in PRIMARY_METRIC.items():
        out = youden_threshold(records, labels, our_key, metric)
        if out is None:
            print(f"  {our_key:<20} {metric:<18} (one class empty in sample)")
            continue
        j, t = out
        print(f"  {our_key:<20} {metric:<18} {t:>11.4f}  {j:>+.2f}")
    print("\n  Compound findings (pneumothorax, consolidation) need joint optimization;")
    print("  inspect their per-finding precision/recall above to guide manual tuning.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labels", required=True, type=Path, help="NIH Data_Entry_*.csv")
    parser.add_argument("--report", type=Path, default=Path("results/report.json"))
    parser.add_argument("--suggest", action="store_true", help="print threshold suggestions via Youden's J")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    records = json.loads(args.report.read_text())

    in_both = [r for r in records if r["image"] in labels]
    skipped = len(records) - len(in_both)
    if not in_both:
        raise SystemExit("No records overlap with the labels CSV — check filenames")
    if skipped:
        print(f"  Note: {skipped} report record(s) had no matching label and were skipped")

    cm = confusion_matrix(in_both, labels)
    print_confusion(cm, len(in_both))
    if args.suggest:
        print_suggestions(in_both, labels)


if __name__ == "__main__":
    main()
