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


@pytest.mark.smoke
@pytest.mark.torch
def test_publication_runner_v1_strict_missing_images_false_drops_rows(
    tmp_path: Path,
) -> None:
    """``strict_missing_images=False`` drops rows whose PNGs are absent on disk.

    Builds a tmp CSV that copies the 16-row fixture and appends 4 rows pointing
    to PNGs that do not exist in the fixture's ``images/`` dir. With the
    permissive flag, the factory must construct the ``NIHDataset`` without
    raising and the resulting dataset must expose only the 16 real rows.
    """
    if not _FIXTURE_CSV.is_file() or not _FIXTURE_IMAGES.is_dir():
        pytest.skip(
            f"NIH fixture not found at {_FIXTURE_ROOT}; "
            f"expected {_FIXTURE_CSV} and {_FIXTURE_IMAGES}"
        )

    fixture_lines = _FIXTURE_CSV.read_text(encoding="utf-8").splitlines()
    # Synthesise 4 extra rows that mimic the CSV schema but reference PNGs
    # that don't exist on disk. Patient IDs intentionally distinct from the
    # fixture's 1..5 so we don't accidentally collide with real rows on
    # patient-grouped operations.
    fake_rows = [
        "99990001_000.png,Cardiomegaly,0,9999,40,M,PA,2500,2500,0.143,0.143",
        "99990001_001.png,Effusion,1,9999,41,M,AP,2500,2500,0.143,0.143",
        "99990002_000.png,No Finding,0,9998,55,F,PA,2500,2500,0.143,0.143",
        "99990002_001.png,Infiltration,1,9998,56,F,AP,2500,2500,0.143,0.143",
    ]
    augmented_csv = tmp_path / "Data_Entry_with_missing.csv"
    augmented_csv.write_text(
        "\n".join([*fixture_lines, *fake_rows]) + "\n", encoding="utf-8"
    )

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    bundle = build_publication_runner_v1(
        seed=0,
        nih_csv_path=augmented_csv,
        nih_images_dir=_FIXTURE_IMAGES,
        artifact_root=artifact_root,
        strict_missing_images=False,
    )

    # The 4 fake rows were dropped; only the 16 fixture rows survive.
    assert len(bundle.dataset.load().samples) == 16


@pytest.mark.smoke
@pytest.mark.torch
def test_publication_runner_v1_feature_cache_dir_populates_npy_files(
    tmp_path: Path,
) -> None:
    """``feature_cache_dir=...`` causes the run to populate per-image .npy files.

    With the cache wired, after one full ``run_experiment`` the cache directory
    contains at least one ``.npy`` file under the backbone-id subtree. This is
    Wave 2's RED-test for Step 3.5: the factory must wrap the inner backbone
    in :class:`CachedBackbone` *before* the ``_DecodingBackbone`` wrapper so
    cache writes happen on the resized 224x224 tensor passed to ResNet50.
    """
    if not _FIXTURE_CSV.is_file() or not _FIXTURE_IMAGES.is_dir():
        pytest.skip(
            f"NIH fixture not found at {_FIXTURE_ROOT}; "
            f"expected {_FIXTURE_CSV} and {_FIXTURE_IMAGES}"
        )

    cache_dir = tmp_path / "feature-cache"
    artifact_root = tmp_path / "artifacts"

    bundle = build_publication_runner_v1(
        seed=0,
        nih_csv_path=_FIXTURE_CSV,
        nih_images_dir=_FIXTURE_IMAGES,
        artifact_root=artifact_root,
        feature_cache_dir=cache_dir,
    )

    run_experiment(
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

    # Cache root exists and contains at least one .npy under the
    # backbone-identifier subtree (CachedBackbone uses ``inner.identifier``
    # as the second path segment).
    assert cache_dir.is_dir()
    npy_files = list(cache_dir.rglob("*.npy"))
    assert len(npy_files) > 0, (
        f"expected the feature cache to be populated under {cache_dir!r}, "
        f"but found no .npy files"
    )


@pytest.mark.smoke
@pytest.mark.torch
def test_publication_runner_v1_txrv_backbone_runs_end_to_end_on_fixture(
    tmp_path: Path,
) -> None:
    """``backbone="txrv-densenet121"`` wires the TXRV CXR-pretrained backbone.

    The factory exposes a ``backbone`` kwarg selecting the embedding source.
    With ``"txrv-densenet121"`` the inner adapter is
    :class:`~harness.adapters.torch.txrv_backbone.TXRVDenseNet121NIHBackbone`
    (DenseNet121, 1024-dim features, NIH-pretrained); the rest of the
    pipeline is unchanged. This smoke test asserts the run completes,
    produces the four v1 artifacts, and records the TXRV identifier in
    the model card's backbone field.
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
        backbone="txrv-densenet121",
    )

    # Config records the TXRV identifier.
    assert bundle.config.backbone_id == "txrv-densenet121-res224-nih"

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

    n_labels = len(result.config.label_names)
    assert n_labels == 14
    assert len(result.report.per_class) == n_labels
    assert (tmp_path / "model_card.json").is_file()
    assert (tmp_path / "thresholds.json").is_file()
    assert (tmp_path / "metric_report.json").is_file()
    assert (tmp_path / "predictions" / "test.csv").is_file()
