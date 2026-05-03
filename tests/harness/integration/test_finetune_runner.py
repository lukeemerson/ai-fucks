"""Integration test for the v1.1 fine-tune runner on the 16-row NIH fixture.

Tiny config (1 epoch, batch=4) -- proves end-to-end wiring between the
:class:`TorchFineTuneTrainer` adapter, the in-memory training dataset
helper, the calibrator + threshold + metrics chain, and the artifact
store. Marked ``@pytest.mark.torch`` so the test is excluded from the
default fast suite (default ``addopts`` is ``-m 'not smoke and not slow
and not torch'``); also ``@pytest.mark.slow`` because even one epoch on
16 fixture rows exercises a real DenseNet121 forward+backward pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.composition.factories import build_finetune_runner_v1
from harness.composition.runner import run_finetune_experiment

_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "harness"
    / "fixtures"
    / "nih"
)
_FIXTURE_CSV = _FIXTURE_ROOT / "Data_Entry_synthetic.csv"
_FIXTURE_IMAGES = _FIXTURE_ROOT / "images"


@pytest.mark.torch
@pytest.mark.slow
def test_finetune_runner_runs_end_to_end_on_fixture(tmp_path: Path) -> None:
    """The v1.1 fine-tune pipeline runs end-to-end on the 16-row NIH fixture.

    This is a wiring smoke test, not a quality test. The 8x8 fixture PNGs
    are upsampled to 224x224 before DenseNet121 and produce uninformative
    features; we only assert the run completes, the report has the right
    shape, and the artifact store wrote its four v1 outputs.
    """
    if not _FIXTURE_CSV.is_file() or not _FIXTURE_IMAGES.is_dir():
        pytest.skip(
            f"NIH fixture not found at {_FIXTURE_ROOT}; "
            f"expected {_FIXTURE_CSV} and {_FIXTURE_IMAGES}"
        )

    bundle = build_finetune_runner_v1(
        seed=0,
        nih_csv_path=_FIXTURE_CSV,
        nih_images_dir=_FIXTURE_IMAGES,
        artifact_root=tmp_path,
        n_epochs=1,
        batch_size=4,
    )

    result = run_finetune_experiment(
        bundle.config,
        dataset=bundle.dataset,
        splitter=bundle.splitter,
        trainer=bundle.trainer,
        calibrator=bundle.calibrator,
        thresholds=bundle.thresholds,
        metrics=bundle.metrics,
        store=bundle.store,
        randomness=bundle.randomness,
        decoder=bundle.decoder,
    )

    n_labels = len(result.config.label_names)
    assert n_labels == 14
    assert len(result.report.per_class) == n_labels
    macro = result.report.macro_f1
    assert 0.0 <= macro.point <= 1.0
    assert macro.lower <= macro.point <= macro.upper

    # Filesystem store wrote the four v1 artifacts.
    assert (tmp_path / "model_card.json").is_file()
    assert (tmp_path / "thresholds.json").is_file()
    assert (tmp_path / "metric_report.json").is_file()
    assert (tmp_path / "predictions" / "test.csv").is_file()

    # Threshold set respects configured clamps.
    cfg_threshold = result.config.threshold
    assert len(result.thresholds.thresholds) == n_labels
    for t in result.thresholds.thresholds:
        assert cfg_threshold.clamp_lo <= t <= cfg_threshold.clamp_hi

    # Model card notes record the trainer identifier.
    assert "torch.finetune.v1" in result.model_card.notes


@pytest.mark.torch
@pytest.mark.slow
def test_finetune_model_card_records_backbone_lineage(tmp_path: Path) -> None:
    """C2: ModelCard.backbone must distinguish densenet121 from resnet50.

    The Wave 4 review found the runner threaded ``trainer.identifier``
    (``"torch.finetune.v1"``) into both ``backbone_id`` and ``head_id``
    slots, losing per-backbone lineage. With the C2 fix the runner uses
    ``config.backbone_id`` (which the factory sets to
    ``f"torch.finetune.{backbone}.v1"``) for the ``backbone`` slot, so
    the persisted model card distinguishes the two backbones.
    """
    if not _FIXTURE_CSV.is_file() or not _FIXTURE_IMAGES.is_dir():
        pytest.skip(
            f"NIH fixture not found at {_FIXTURE_ROOT}; "
            f"expected {_FIXTURE_CSV} and {_FIXTURE_IMAGES}"
        )

    backbones: dict[str, str] = {}
    for backbone in ("densenet121", "resnet50"):
        run_root = tmp_path / backbone
        run_root.mkdir()
        bundle = build_finetune_runner_v1(
            seed=0,
            nih_csv_path=_FIXTURE_CSV,
            nih_images_dir=_FIXTURE_IMAGES,
            artifact_root=run_root,
            n_epochs=1,
            batch_size=4,
            backbone=backbone,
        )
        result = run_finetune_experiment(
            bundle.config,
            dataset=bundle.dataset,
            splitter=bundle.splitter,
            trainer=bundle.trainer,
            calibrator=bundle.calibrator,
            thresholds=bundle.thresholds,
            metrics=bundle.metrics,
            store=bundle.store,
            randomness=bundle.randomness,
            decoder=bundle.decoder,
        )
        backbones[backbone] = result.model_card.backbone

    assert backbones["densenet121"] != backbones["resnet50"], (
        f"ModelCard.backbone must distinguish backbones, got {backbones!r}"
    )
    assert "densenet121" in backbones["densenet121"]
    assert "resnet50" in backbones["resnet50"]
