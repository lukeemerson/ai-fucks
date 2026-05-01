"""Integration test exercising the sklearn-backed adapter set.

We still use the in-memory artifact store; the goal is to verify the runner
composes the production sklearn adapters correctly, not to exercise on-disk
artifact persistence.
"""

from __future__ import annotations

from harness.composition.factories import build_v1_runner_sklearn
from harness.composition.runner import run_experiment
from harness.domain.types import ExperimentResult


def _run_sklearn(seed: int) -> ExperimentResult:
    bundle = build_v1_runner_sklearn(seed)
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


def test_runner_completes_with_sklearn_adapters() -> None:
    result = _run_sklearn(seed=0)
    n_labels = len(result.config.label_names)

    # Per-class metrics shape is well-formed.
    assert len(result.report.per_class) == n_labels
    for entry in result.report.per_class:
        assert 0.0 <= entry.f1.lower <= entry.f1.point <= entry.f1.upper <= 1.0
        assert 0.0 <= entry.auroc.lower <= entry.auroc.point <= entry.auroc.upper <= 1.0
        assert entry.support >= 0

    # Macro intervals are non-degenerate (point inside CI).
    macro = result.report.macro_f1
    assert macro.lower <= macro.point <= macro.upper

    # Thresholds respect configured clamps per class.
    cfg_threshold = result.config.threshold
    assert len(result.thresholds.thresholds) == n_labels
    for t in result.thresholds.thresholds:
        assert cfg_threshold.clamp_lo <= t <= cfg_threshold.clamp_hi

    # Test predictions / probabilities align in shape and order.
    assert (
        result.test_predictions.sample_ids
        == result.test_probabilities.sample_ids
    )
    assert result.test_predictions.values.shape[1] == n_labels
