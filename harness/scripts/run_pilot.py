"""Pilot CLI for the v1 publication pipeline (PAPER_CHECKLIST.md Step 3).

Wires :func:`harness.composition.factories.build_publication_runner_v1`
against an on-disk NIH ChestX-ray14 manifest, runs
:func:`harness.composition.runner.run_experiment`, and prints a one-shot
summary plus the artifact paths.

Usage::

    python -m harness.scripts.run_pilot \\
        --nih-csv /path/to/Data_Entry_2017_v2020.csv \\
        --nih-images /path/to/images \\
        --n 10000

Image dtype/range note
----------------------
:class:`~harness.adapters.fs.nih_dataset.NIHDataset.get_image_bytes` returns
**raw on-disk PNG bytes** (not a numpy array). Decoding into the
``[0, 1] float32`` tensor that ``TorchVisionResNet50Backbone`` expects
happens inside the composition wiring (see
``harness/composition/_publication_pipeline.py``); the script is therefore
free of any image-conversion logic. The ``--n`` subsampler operates on
``Sample`` records, not pixel arrays.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from harness.composition.factories import build_publication_runner_v1
from harness.composition.runner import run_experiment
from harness.domain.types import ExperimentResult

__all__ = ["main"]


def _nonneg_int(raw: str) -> int:
    """argparse type validator: accept non-negative integers only.

    ``--n`` is a sample-count cap; negative values have no defined semantics
    so we reject them at parse time rather than coercing silently.
    """
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"--n must be >= 0, got {value}"
        )
    return value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness.scripts.run_pilot",
        description=(
            "Pilot run of the v1 publication pipeline on a slice of the NIH "
            "ChestX-ray14 dataset."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed for the run (default: 0).",
    )
    parser.add_argument(
        "--nih-csv",
        type=Path,
        required=True,
        help="Path to Data_Entry_2017_v2020.csv (or a fixture CSV).",
    )
    parser.add_argument(
        "--nih-images",
        type=Path,
        required=True,
        help="Path to the flat directory of NIH PNGs.",
    )
    parser.add_argument(
        "--n",
        type=_nonneg_int,
        default=10000,
        help=(
            "First-N samples in CSV order (0 = full dataset). "
            "Patient-leakage-free (the NIH CSV is grouped by patient), but "
            "the truncation may end mid-patient. Patient-block-aligned "
            "truncation is deferred to v1.1 ablations. (default: 10000)"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Filesystem root for artifacts. Defaults to "
            "``runs/<UTC-timestamp>/`` under the current working directory."
        ),
    )
    parser.add_argument(
        "--strict-missing-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set, the NIH dataset adapter raises DataError when CSV rows "
            "reference missing PNGs. Use --no-strict-missing-images for "
            "partial dataset dumps (e.g. the public 5k sample). Default: on."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_summary(
    result: ExperimentResult, artifact_root: Path
) -> str:
    """Single-block textual summary printed at end of run."""
    n_labels = len(result.config.label_names)
    macro = result.report.macro_f1
    lines = [
        "",
        "=" * 70,
        f"experiment: {result.config.experiment_name}",
        f"seed:       {result.config.seed}",
        f"split:      train={len(result.split.train_indices)}, "
        f"val={len(result.split.val_indices)}, "
        f"test={len(result.split.test_indices)}",
        f"labels:     {n_labels}",
        "",
        f"macro-F1:   {macro.point:.4f}  "
        f"(95% CI: {macro.lower:.4f}, {macro.upper:.4f})",
        f"macro-AUROC: {result.report.macro_auroc.point:.4f}",
        f"macro-AUPRC: {result.report.macro_auprc.point:.4f}",
        "",
        "per-class F1:",
    ]
    for entry in result.report.per_class:
        lines.append(
            f"  {entry.label:<22s}  F1={entry.f1.point:.4f} "
            f"AUROC={entry.auroc.point:.4f}  support={entry.support}"
        )
    lines += [
        "",
        f"artifacts written to: {artifact_root.resolve()}",
        "=" * 70,
    ]
    return "\n".join(lines)


def _resolve_artifact_root(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / timestamp


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    artifact_root = _resolve_artifact_root(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    n_samples: int | None = args.n if args.n > 0 else None
    bundle = build_publication_runner_v1(
        seed=args.seed,
        nih_csv_path=args.nih_csv,
        nih_images_dir=args.nih_images,
        artifact_root=artifact_root,
        n_samples=n_samples,
        strict_missing_images=args.strict_missing_images,
    )
    if n_samples is not None:
        actual = len(bundle.dataset.load().samples)
        if actual < n_samples:
            print(
                f"[pilot] warning: --n={n_samples} exceeds dataset size "
                f"({actual}); using full dataset",
                file=sys.stderr,
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

    print(_format_summary(result, artifact_root))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
