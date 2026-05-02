"""Ablation CLI for the v1 publication pipeline (PAPER_CHECKLIST.md Step 3.5).

Runs :func:`harness.composition.runner.run_experiment` once per seed against a
shared :class:`~harness.adapters.fs.cached_backbone.CachedBackbone` feature
cache, then writes a comparison CSV summarising macro metrics across the seed
grid. The cache is the whole point: feature extraction is by far the most
expensive step, and seed-only variation does not change image bytes, so every
seed after the first hits the cache for every row.

v1 variant axis = master seed only. Future axes (head, calibrator, threshold)
are deferred to v1.1 -- when they land, the runner grows additional
``--variant-*`` flags rather than a new entry point.

Usage::

    python -m harness.scripts.run_ablation \\
        --seeds 0,1,2,3,4 \\
        --nih-csv /path/to/Data_Entry_2017_v2020.csv \\
        --nih-images /path/to/images \\
        --feature-cache-dir /path/to/feature-cache \\
        --artifact-root runs/ablation-2026-05-01

Failure handling
----------------
If a single seed raises mid-run, the exception is captured, the seed is logged
to stderr, and the loop continues with the remaining seeds. The script exits
with status ``1`` if any seed failed (and ``0`` otherwise). ``comparison.csv``
includes only the seeds that completed successfully -- failures do not appear
as partial rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from harness.composition.factories import build_publication_runner_v1
from harness.composition.runner import run_experiment
from harness.domain.types import ExperimentResult

__all__ = ["main"]


# Comparison-CSV column order. Kept stable so downstream consumers can rely on
# the layout; new columns are added on the right in v1.1.
_COMPARISON_COLUMNS: tuple[str, ...] = (
    "seed",
    "macro_f1",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "macro_auroc",
    "macro_auprc",
)


@dataclass(frozen=True, slots=True)
class _SeedRow:
    """One successful variant's headline metrics, ready for CSV emission."""

    seed: int
    macro_f1: float
    macro_f1_ci_low: float
    macro_f1_ci_high: float
    macro_auroc: float
    macro_auprc: float

    def as_csv_row(self) -> tuple[str, ...]:
        return (
            str(self.seed),
            f"{self.macro_f1:.6f}",
            f"{self.macro_f1_ci_low:.6f}",
            f"{self.macro_f1_ci_high:.6f}",
            f"{self.macro_auroc:.6f}",
            f"{self.macro_auprc:.6f}",
        )


def _parse_seeds(raw: str) -> tuple[int, ...]:
    """Parse ``--seeds`` (comma-separated ints), reject empty / duplicates.

    Empty list and duplicates are configuration mistakes (an ablation with
    zero or repeated seeds is not a meaningful comparison), so we surface
    them at parse time rather than coercing silently.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "--seeds must be a non-empty comma-separated list of integers, "
            f"got {raw!r}"
        )
    seeds: list[int] = []
    for p in parts:
        try:
            seeds.append(int(p))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--seeds entry {p!r} is not an integer"
            ) from exc
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            f"--seeds must not contain duplicates, got {seeds}"
        )
    return tuple(seeds)


def _nonneg_int(raw: str) -> int:
    """argparse type validator: accept non-negative integers only.

    Mirrors :func:`harness.scripts.run_pilot._nonneg_int` so the two CLIs
    have the same ``--n`` contract (``0`` = full dataset, negative rejected).
    """
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"--n must be >= 0, got {value}"
        )
    return value


def _resolve_artifact_root(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"ablation-{timestamp}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness.scripts.run_ablation",
        description=(
            "Run the v1 publication pipeline across a seed grid and write a "
            "comparison CSV. Uses a shared CachedBackbone feature cache so "
            "feature extraction runs once per unique input image."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        required=True,
        help=(
            "Comma-separated list of master seeds (e.g. ``0,1,2,3,4``). "
            "Must be non-empty; duplicates are rejected."
        ),
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
        default=0,
        help=(
            "First-N samples in CSV order (0 = full dataset; default 0). "
            "Patient-leakage-free but may end mid-patient. Mirrors "
            "``run_pilot.py --n``."
        ),
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        required=True,
        help=(
            "Filesystem root for the shared feature cache. Created if it "
            "does not exist. The cache is content-addressable, so reusing "
            "the same directory across ablation runs accelerates everything "
            "downstream of feature extraction."
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
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Filesystem root for per-seed artifacts. Defaults to "
            "``runs/ablation-<UTC-timestamp>/``. Each seed lands under "
            "``<artifact-root>/seed-<n>/``; the comparison CSV lands at "
            "``<artifact-root>/comparison.csv``."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _row_from_result(seed: int, result: ExperimentResult) -> _SeedRow:
    f1 = result.report.macro_f1
    return _SeedRow(
        seed=seed,
        macro_f1=float(f1.point),
        macro_f1_ci_low=float(f1.lower),
        macro_f1_ci_high=float(f1.upper),
        macro_auroc=float(result.report.macro_auroc.point),
        macro_auprc=float(result.report.macro_auprc.point),
    )


def _write_comparison_csv(rows: Sequence[_SeedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_COMPARISON_COLUMNS)
        for row in rows:
            writer.writerow(row.as_csv_row())


def _format_summary(
    rows: Sequence[_SeedRow],
    *,
    feature_cache_dir: Path,
    n_variants: int,
    n_failed: int,
) -> str:
    lines = [
        "",
        "=" * 78,
        "ablation: seed-grid",
        f"feature cache: {feature_cache_dir.resolve()}",
        f"variants: {n_variants} (succeeded: {len(rows)}, failed: {n_failed})",
        "",
        f"{'seed':<6}{'macro-F1':<12}{'95% CI':<26}"
        f"{'macro-AUROC':<14}{'macro-AUPRC':<12}",
    ]
    for row in rows:
        ci = f"[{row.macro_f1_ci_low:.4f}, {row.macro_f1_ci_high:.4f}]"
        lines.append(
            f"{row.seed:<6d}"
            f"{row.macro_f1:<12.4f}"
            f"{ci:<26s}"
            f"{row.macro_auroc:<14.4f}"
            f"{row.macro_auprc:<12.4f}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run_one_seed(
    seed: int,
    *,
    nih_csv: Path,
    nih_images: Path,
    artifact_root: Path,
    n_samples: int | None,
    strict_missing_images: bool,
    feature_cache_dir: Path,
) -> ExperimentResult:
    """Build the bundle for ``seed`` and run the experiment."""
    bundle = build_publication_runner_v1(
        seed=seed,
        nih_csv_path=nih_csv,
        nih_images_dir=nih_images,
        artifact_root=artifact_root,
        n_samples=n_samples,
        strict_missing_images=strict_missing_images,
        feature_cache_dir=feature_cache_dir,
    )
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    seeds: tuple[int, ...] = args.seeds
    artifact_root = _resolve_artifact_root(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    n_samples: int | None = args.n if args.n > 0 else None

    rows: list[_SeedRow] = []
    n_failed = 0

    for seed in seeds:
        per_seed_root = artifact_root / f"seed-{seed}"
        per_seed_root.mkdir(parents=True, exist_ok=True)
        try:
            result = _run_one_seed(
                seed,
                nih_csv=args.nih_csv,
                nih_images=args.nih_images,
                artifact_root=per_seed_root,
                n_samples=n_samples,
                strict_missing_images=args.strict_missing_images,
                feature_cache_dir=args.feature_cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 -- per-seed failures must not abort
            # Surface the failure to stderr verbatim (no swallowing) but keep
            # the loop running so the remaining seeds still get a chance.
            print(f"seed={seed} failed: {exc!r}", file=sys.stderr)
            n_failed += 1
            continue
        rows.append(_row_from_result(seed, result))

    comparison_path = artifact_root / "comparison.csv"
    _write_comparison_csv(rows, comparison_path)

    print(
        _format_summary(
            rows,
            feature_cache_dir=args.feature_cache_dir,
            n_variants=len(seeds),
            n_failed=n_failed,
        )
    )
    print(f"comparison written to: {comparison_path.resolve()}")
    return 1 if n_failed else 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
