"""Joint grid-search tuner for compound-rule thresholds.

The single-metric findings (cardiomegaly, pleural_effusion, pulmonary_edema,
atelectasis, emphysema, focal_opacity) are tuned via Youden's J in
``analyzer.evaluate --suggest`` — one metric, one cutoff, one ROC sweep.

The compound rules are not separable that way. They combine multiple metrics
with AND clauses, so the F1-optimal cutoff for one metric depends on the
cutoffs chosen for the others. This module does a coarse joint grid search
per compound rule on a held-out 80/20 train/test split (deterministic by
image-filename hash, so reports can be regenerated without leaking).

Usage:
    python -m analyzer.tune --labels Data_Entry_2017_v2020.csv

Outputs the best ``(threshold tuple, train F1, test F1)`` per compound rule.
The caller decides whether to copy the suggestion into ``main.THRESHOLDS``;
this module never mutates source files.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyzer.evaluate import Confusion, is_positive, load_labels

# A report record after JSON-deserialisation. We only ever read `image`
# (str) and `metrics` (dict[str, float]) here, but other keys (`findings`)
# may be present, so leave the value side as Any.
Record = dict[str, Any]
# Metric dict: {metric_name: value}. Predicates close over this shape.
Metrics = dict[str, float]
# Predicate over a metrics dict.
Predicate = Callable[[Metrics], bool]


# ---------------------------------------------------------------------------
# Compound rule definitions.
#
# Each entry describes a compound finding, the predicate factory that turns a
# tuple of cutoffs into a `metrics -> bool` function, and the grid of cutoff
# tuples to search. Grid resolution is deliberately coarse (~10 values per
# axis) — chest-X-ray pixel statistics are noisy enough that finer sweeps
# overfit the train split. Total candidates per rule ≈ 10**axes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompoundRule:
    name: str  # finding key
    axes: tuple[str, ...]  # axis labels for printing
    grid: tuple[tuple[float, ...], ...]  # one tuple of values per axis
    predicate: Callable[[tuple[float, ...]], Predicate]


def _ptx_predicate(cutoffs: tuple[float, ...]) -> Predicate:
    mean_cut, std_cut = cutoffs

    def test(f: Metrics) -> bool:
        return (f["ptx_left_mean"] < mean_cut and f["ptx_left_std"] < std_cut) or (
            f["ptx_right_mean"] < mean_cut and f["ptx_right_std"] < std_cut
        )

    return test


def _consolidation_predicate(cutoffs: tuple[float, ...]) -> Predicate:
    haze_cut, basal_cut, focal_cut = cutoffs

    def test(f: Metrics) -> bool:
        return (
            f["bilateral_haze"] > haze_cut
            and f["basal_opacity"] > basal_cut
            and f["focal_variance"] < focal_cut
        )

    return test


def _linspace(lo: float, hi: float, n: int) -> tuple[float, ...]:
    if n < 2:
        return (lo,)
    step = (hi - lo) / (n - 1)
    return tuple(round(lo + step * i, 4) for i in range(n))


COMPOUND_RULES: tuple[CompoundRule, ...] = (
    CompoundRule(
        name="pneumothorax",
        axes=("ptx_*_mean", "ptx_*_std"),
        # Peripheral mean/std on 0-1 grayscale; current shipped cutoffs are
        # (0.28, 0.08). Search a band around them.
        grid=(_linspace(0.18, 0.42, 9), _linspace(0.04, 0.14, 6)),
        predicate=_ptx_predicate,
    ),
    CompoundRule(
        name="consolidation",
        axes=("bilateral_haze", "basal_opacity", "focal_variance"),
        # Current cutoffs (0.48, 0.48, 0.050). Haze + basal are 0-1 means;
        # focal_variance is small (often < 0.10).
        grid=(
            _linspace(0.35, 0.65, 7),
            _linspace(0.35, 0.65, 7),
            _linspace(0.020, 0.080, 7),
        ),
        predicate=_consolidation_predicate,
    ),
)


# ---------------------------------------------------------------------------
# Train/test split: deterministic per filename so the same image always lands
# in the same bucket regardless of the order in which reports are merged.
# We hash with sha1 (stable across Python versions) and bucket by comparing
# the first byte of the digest against ``round(test_frac * 256)``: bytes
# strictly less than that threshold land in the test bucket, the rest in
# train. With ``test_frac=0.2`` the threshold is 51, giving roughly a 20/80
# split assuming a uniform hash distribution.
# ---------------------------------------------------------------------------


def split_train_test(records: Sequence[Record], test_frac: float = 0.2) -> tuple[list[Record], list[Record]]:
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1); got {test_frac}")
    train: list[Record] = []
    test: list[Record] = []
    threshold = int(round(test_frac * 256))
    for r in records:
        h = hashlib.sha1(r["image"].encode("utf-8")).digest()[0]
        (test if h < threshold else train).append(r)
    return train, test


# ---------------------------------------------------------------------------
# F1 scoring under a candidate predicate.
# ---------------------------------------------------------------------------


def _score(
    records: Iterable[Record],
    labels: dict[str, set[str]],
    our_key: str,
    predicate: Predicate,
) -> Confusion:
    cm = Confusion()
    for r in records:
        nih = labels.get(r["image"])
        if nih is None:
            continue
        pred = predicate(r["metrics"])
        actual = is_positive(nih, our_key)
        if pred and actual:
            cm.tp += 1
        elif pred and not actual:
            cm.fp += 1
        elif not pred and actual:
            cm.fn += 1
        else:
            cm.tn += 1
    return cm


def _iter_candidates(rule: CompoundRule) -> Iterable[tuple[float, ...]]:
    return itertools.product(*rule.grid)


@dataclass
class TuningResult:
    rule: CompoundRule
    best_cutoffs: tuple[float, ...]
    train_f1: float
    test_f1: float
    train_cm: Confusion
    test_cm: Confusion
    n_train_pos: int
    n_test_pos: int


def tune_compound(
    rule: CompoundRule,
    train: Sequence[Record],
    test: Sequence[Record],
    labels: dict[str, set[str]],
) -> TuningResult | None:
    """Grid-search the rule's cutoffs on ``train`` and score on ``test``.

    Returns ``None`` (so the caller can warn and skip) when:

    * the train split has zero positives — nothing to fit against;
    * the test split has zero positives — F1 on the held-out set is
      undefined (every positive prediction is a false positive), so we
      can't tell whether a candidate generalises;
    * no candidate beats F1=0 on the train split — picking the
      lexicographically-first tied candidate would silently ship a
      degenerate threshold. Tuning is only useful when it finds a
      genuinely better cutoff than the trivial "predict-nothing" rule.
    """
    n_train_pos = sum(1 for r in train if is_positive(labels.get(r["image"], set()), rule.name))
    n_test_pos = sum(1 for r in test if is_positive(labels.get(r["image"], set()), rule.name))
    if n_train_pos == 0:
        return None
    if n_test_pos == 0:
        return None

    best_f1: float = -1.0
    best_cut: tuple[float, ...] = tuple(axis[0] for axis in rule.grid)
    best_train_cm = Confusion()
    for cutoffs in _iter_candidates(rule):
        cm = _score(train, labels, rule.name, rule.predicate(cutoffs))
        if cm.f1 > best_f1:
            best_f1, best_cut, best_train_cm = cm.f1, cutoffs, cm

    # Refuse to declare a winner if every candidate tied at F1=0 — the point
    # of tuning is to find a *better* cutoff, not the first one alphabetically.
    if best_f1 <= 0.0:
        return None

    test_cm = _score(test, labels, rule.name, rule.predicate(best_cut))
    return TuningResult(
        rule=rule,
        best_cutoffs=best_cut,
        train_f1=best_f1,
        test_f1=test_cm.f1,
        train_cm=best_train_cm,
        test_cm=test_cm,
        n_train_pos=n_train_pos,
        n_test_pos=n_test_pos,
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _print_result(res: TuningResult) -> None:
    rule = res.rule
    print(f"\n  {rule.name}  (positives: train={res.n_train_pos}, test={res.n_test_pos})")
    axes_str = ", ".join(rule.axes)
    cuts_str = ", ".join(f"{c:.4f}" for c in res.best_cutoffs)
    print(f"    best cutoffs ({axes_str}) = ({cuts_str})")
    print(f"    train  P={res.train_cm.precision:.2f} R={res.train_cm.recall:.2f} F1={res.train_f1:.3f}")
    print(f"    test   P={res.test_cm.precision:.2f} R={res.test_cm.recall:.2f} F1={res.test_f1:.3f}")


def _skip_reason(
    rule: CompoundRule,
    train: Sequence[Record],
    test: Sequence[Record],
    labels: dict[str, set[str]],
) -> str:
    """Re-derive the reason ``tune_compound`` returned ``None``.

    Cheap to recompute on the skip path and keeps the warning specific.
    """
    n_train_pos = sum(1 for r in train if is_positive(labels.get(r["image"], set()), rule.name))
    n_test_pos = sum(1 for r in test if is_positive(labels.get(r["image"], set()), rule.name))
    if n_train_pos == 0:
        return "zero positives in train split"
    if n_test_pos == 0:
        return "test split has no positives"
    return "no candidate beat F1=0 on the train split"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labels", required=True, type=Path, help="NIH Data_Entry_*.csv")
    parser.add_argument("--report", type=Path, default=Path("results/report.json"))
    parser.add_argument("--test-frac", type=float, default=0.2, help="held-out test fraction (default 0.2)")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    records = json.loads(args.report.read_text())
    in_both = [r for r in records if r["image"] in labels]
    if not in_both:
        raise SystemExit("No records overlap with the labels CSV — check filenames")

    train, test = split_train_test(in_both, args.test_frac)
    print(
        f"  Joint grid search — {len(in_both)} labeled records "
        f"({len(train)} train / {len(test)} test, hash-bucketed)"
    )

    for rule in COMPOUND_RULES:
        res = tune_compound(rule, train, test, labels)
        if res is None:
            reason = _skip_reason(rule, train, test, labels)
            print(f"\n  {rule.name}: {reason} for {rule.name}; skipping")
            continue
        _print_result(res)

    print(
        "\n  Note: cutoffs are NOT auto-applied. Copy them into "
        "analyzer.main.THRESHOLDS by hand, regenerate fixtures/seed_report.json, "
        "then update tests/test_findings.py to pin the new values."
    )


if __name__ == "__main__":
    main()
