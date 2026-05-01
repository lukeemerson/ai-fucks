"""Golden-path integration tests for ``run_experiment`` with fakes."""

from __future__ import annotations

from harness.composition.factories import build_v1_runner_with_fakes
from harness.composition.runner import run_experiment
from harness.domain.types import ExperimentResult

_EXPECTED_ARTIFACT_KEYS = {
    "model_card",
    "thresholds",
    "test_predictions",
    "metric_report",
}


def _run_with_fakes(seed: int) -> ExperimentResult:
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


def test_runner_with_fakes_returns_well_formed_result() -> None:
    result = _run_with_fakes(seed=42)
    n_labels = len(result.config.label_names)

    # Per-class entries match label vocabulary.
    assert len(result.report.per_class) == n_labels
    assert tuple(c.label for c in result.report.per_class) == result.config.label_names

    # Threshold set matches the label vocabulary.
    assert len(result.thresholds.thresholds) == n_labels
    assert result.thresholds.label_names == result.config.label_names

    # Probabilities and predictions agree on test size.
    assert len(result.test_probabilities.sample_ids) == len(result.split.test_indices)
    assert (
        result.test_predictions.sample_ids == result.test_probabilities.sample_ids
    )
    assert result.test_predictions.values.shape == (
        len(result.split.test_indices),
        n_labels,
    )


def test_runner_writes_expected_artifact_keys() -> None:
    result = _run_with_fakes(seed=42)
    assert set(result.artifact_uris) == _EXPECTED_ARTIFACT_KEYS
    for uri in result.artifact_uris.values():
        assert uri.startswith("memory://")


def test_runner_populates_model_card() -> None:
    result = _run_with_fakes(seed=42)
    card = result.model_card
    assert card.name == result.config.experiment_name
    assert card.label_names == result.config.label_names
    assert card.train_size == len(result.split.train_indices)
    assert card.val_size == len(result.split.val_indices)
    assert card.test_size == len(result.split.test_indices)
    assert card.config_hash  # non-empty
    assert card.threshold_method == result.thresholds.method
