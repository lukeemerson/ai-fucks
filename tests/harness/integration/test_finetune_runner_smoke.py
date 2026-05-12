"""Load-bearing smoke test for the v1.1 fine-tune runner.

Per FINE_TUNING_DESIGN.md §6.4 (the approved §10 answer #1): build the
fine-tune runner against the real 4999-sample NIH slice, train for 1
epoch, and assert ``report.macro_auroc.point > 0.65`` -- the
frozen-feature floor the existing pipeline achieved on this slice. One
epoch of fine-tuning on 4k samples should at minimum match this; failure
to clear it signals a regression in the trainer adapter and the PR must
not merge.

Marked ``smoke`` AND ``slow`` AND ``torch`` -- excluded from the default
fast suite by every marker; opt in with ``pytest -m smoke``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.composition.factories import build_finetune_runner_v1
from harness.composition.runner import run_finetune_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REAL_CSV = PROJECT_ROOT / "Data_Entry_2017_v2020.csv"
REAL_IMAGES = PROJECT_ROOT / "db-test_images" / "images"


pytestmark = [
    pytest.mark.smoke,
    pytest.mark.slow,
    pytest.mark.torch,
    pytest.mark.skipif(
        not REAL_CSV.is_file() or not REAL_IMAGES.is_dir(),
        reason=(
            f"local NIH data not available "
            f"(expected {REAL_CSV} and {REAL_IMAGES})"
        ),
    ),
]


def test_finetune_smoke_macro_auroc_above_065(tmp_path: Path) -> None:
    """1 epoch of fine-tuning on the 4999-row slice clears macro-AUROC > 0.65.

    Per the §10 sign-off the smoke gate is the load-bearing quality check
    for the v1.1 fine-tune adapter.
    """
    bundle = build_finetune_runner_v1(
        seed=0,
        nih_csv_path=REAL_CSV,
        nih_images_dir=REAL_IMAGES,
        artifact_root=tmp_path,
        strict_missing_images=False,
        n_epochs=1,
        batch_size=32,
        learning_rate=1e-4,
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

    macro_auroc = result.report.macro_auroc.point
    print(f"\n[smoke] macro_auroc.point = {macro_auroc:.4f}")
    assert macro_auroc > 0.65, (
        f"macro-AUROC = {macro_auroc:.4f} <= 0.65; trainer regression "
        "(FINE_TUNING_DESIGN.md §6.4)"
    )
