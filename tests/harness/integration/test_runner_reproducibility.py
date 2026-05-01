"""Reproducibility tests for ``run_experiment`` with the fake adapter set."""

from __future__ import annotations

import numpy as np

from harness.composition.factories import build_v1_runner_with_fakes
from harness.composition.runner import run_experiment
from harness.domain.types import ExperimentResult


def _run(seed: int) -> ExperimentResult:
    bundle = build_v1_runner_with_fakes(seed)
    return run_experiment(
        bundle.config,
        dataset=bundle.dataset,
        splitter=bundle.splitter,
        backbone=bundle.backbone,
        head=bundle.head,
        calibrator=bundle.calibrator,
        thresholds=bundle.thresholds,
        metrics=bundle.metrics,
        store=bundle.store,
        randomness=bundle.randomness,
    )


def test_same_seed_yields_identical_metrics_and_thresholds() -> None:
    a = _run(seed=42)
    b = _run(seed=42)

    # Per-class thresholds match exactly.
    assert a.thresholds.thresholds == b.thresholds.thresholds
    assert a.thresholds.label_names == b.thresholds.label_names

    # Test predictions are byte-identical for the same seed.
    np.testing.assert_array_equal(
        a.test_predictions.values, b.test_predictions.values
    )

    # Macro metrics: point AND CI bounds are byte-identical.
    for macro_a, macro_b in (
        (a.report.macro_f1, b.report.macro_f1),
        (a.report.macro_auroc, b.report.macro_auroc),
        (a.report.macro_auprc, b.report.macro_auprc),
    ):
        assert macro_a.point == macro_b.point
        assert macro_a.lower == macro_b.lower
        assert macro_a.upper == macro_b.upper

    # Per-class metrics: point AND CI bounds (low, high) match exactly.
    assert len(a.report.per_class) == len(b.report.per_class)
    for cls_a, cls_b in zip(a.report.per_class, b.report.per_class, strict=True):
        assert cls_a.label == cls_b.label
        assert cls_a.support == cls_b.support
        for ia, ib in (
            (cls_a.f1, cls_b.f1),
            (cls_a.auroc, cls_b.auroc),
            (cls_a.auprc, cls_b.auprc),
        ):
            assert ia.point == ib.point
            assert ia.lower == ib.lower
            assert ia.upper == ib.upper

    # Stable model-card identifier: prefer config_hash if present, fall back to
    # the threshold-method/shrinkage/clamp tuple as a structural proxy. Skip
    # ``created_at`` -- a separate fix pins it to epoch when clock=None and the
    # field is not part of the reproducibility contract here.
    assert a.model_card.config_hash == b.model_card.config_hash
    assert a.model_card.threshold_method == b.model_card.threshold_method
    assert a.thresholds.shrinkage == b.thresholds.shrinkage
    assert a.thresholds.clamp_lo == b.thresholds.clamp_lo
    assert a.thresholds.clamp_hi == b.thresholds.clamp_hi


def test_different_seeds_change_at_least_one_output() -> None:
    a = _run(seed=42)
    b = _run(seed=7)

    differs = (
        a.thresholds.thresholds != b.thresholds.thresholds
        or a.report.macro_f1.point != b.report.macro_f1.point
        or a.split.train_indices != b.split.train_indices
    )
    assert differs, "expected different seeds to change at least one output"
