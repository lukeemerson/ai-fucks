"""Smoke test for the v1 publication composition.

Wires the full pipeline -- ``NIHDataset`` (real PNG bytes) ->
``IterativeStratifiedPatientSplitter`` -> ``TorchVisionResNet50Backbone``
(ImageNet weights with random init for offline tests) -> linear classifier
head -> ``PerClassIsotonicCalibrator`` -> ``PrSweepShrinkageThreshold`` ->
``BootstrapMetrics`` -> ``FilesystemArtifactStore`` -- and asserts the run
produces a non-degenerate :class:`ExperimentResult` and writes the four v1
artifacts to disk.

Marked ``smoke`` AND ``torch`` because (1) it touches a real on-disk fixture
and (2) it imports the real torchvision adapter. Both markers are excluded
from the default fast suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.composition.factories import build_publication_runner_v1
from harness.composition.runner import run_experiment

# Resolve the 16-row NIH fixture from the repo root. ``parents[3]`` is
# tests/harness/integration -> tests/harness -> tests -> repo root.
_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "harness"
    / "fixtures"
    / "nih"
)
_FIXTURE_CSV = _FIXTURE_ROOT / "Data_Entry_synthetic.csv"
_FIXTURE_IMAGES = _FIXTURE_ROOT / "images"


@pytest.mark.smoke
@pytest.mark.torch
def test_publication_runner_v1_runs_end_to_end_on_fixture(
    tmp_path: Path,
) -> None:
    """The v1 publication pipeline runs end-to-end on the 16-row NIH fixture.

    This is a wiring smoke test, not a quality test. The 8x8 fixture PNGs are
    upsampled to 224x224 before the ResNet stem and produce uninformative
    features; we only assert the run completes, the report has the right
    shape, and the artifact store wrote its four v1 outputs.
    """
    if not _FIXTURE_CSV.is_file() or not _FIXTURE_IMAGES.is_dir():
        pytest.skip(
            f"NIH fixture not found at {_FIXTURE_ROOT}; "
            f"expected {_FIXTURE_CSV} and {_FIXTURE_IMAGES}"
        )

    bundle = build_publication_runner_v1(
        seed=0,
        nih_csv_path=_FIXTURE_CSV,
        nih_images_dir=_FIXTURE_IMAGES,
        artifact_root=tmp_path,
    )

    result = run_experiment(
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

    # Report has one entry per NIH-14 label.
    n_labels = len(result.config.label_names)
    assert n_labels == 14
    assert len(result.report.per_class) == n_labels

    # Macro-F1 is a real number in [0, 1] -- "any number is fine; we're just
    # proving wiring" per the agent brief.
    macro = result.report.macro_f1
    assert 0.0 <= macro.point <= 1.0
    assert macro.lower <= macro.point <= macro.upper

    # Every patient ended up in exactly one split (patient-level
    # stratification contract). The fixture has 5 patients; with val=test=0.2
    # we should at least get one train patient.
    assert len(result.split.train_indices) >= 1

    # The filesystem store wrote the four v1 artifacts.
    assert (tmp_path / "model_card.json").is_file()
    assert (tmp_path / "thresholds.json").is_file()
    assert (tmp_path / "metric_report.json").is_file()
    assert (tmp_path / "predictions" / "test.csv").is_file()

    # Threshold set respects configured clamps.
    cfg_threshold = result.config.threshold
    assert len(result.thresholds.thresholds) == n_labels
    for t in result.thresholds.thresholds:
        assert cfg_threshold.clamp_lo <= t <= cfg_threshold.clamp_hi
