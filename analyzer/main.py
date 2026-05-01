from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from analyzer.features import extract_features
from analyzer.predict import (
    DEFAULT_THRESHOLD_RULES,
    build_report_record,
    detect_findings,
    format_report,
    threshold_predictions,
)
from analyzer.profile import load_profile, predict_with_profile

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "db-test_images" / "images"
DASHBOARD_TMPL = Path(__file__).parent / "dashboard.html"
THRESHOLDS = DEFAULT_THRESHOLD_RULES
__all__ = ["THRESHOLDS", "detect_findings", "main"]


def summary_table(results: list[tuple[str, list[str]]]) -> str:
    all_keys = list(THRESHOLDS.keys())
    header = f"\n{'─' * 60}\n  SUMMARY — {len(results)} Images\n{'─' * 60}"
    col_w = 18
    row = f"  {'Image':<28}" + "".join(f"{k[:col_w]:<{col_w}}" for k in all_keys)
    lines = [header, row, "  " + "─" * (28 + col_w * len(all_keys))]
    for name, found in results:
        short = name[:26]
        cells = "".join(("  ✓  " if k in found else "  -  ").ljust(col_w) for k in all_keys)
        lines.append(f"  {short:<28}{cells}")
    return "\n".join(lines)


def _iter_images(args: argparse.Namespace) -> list[Path]:
    if args.image is not None:
        return [args.image]

    images_dir = args.images_dir
    all_images = sorted(images_dir.glob("*.png"))
    batch = all_images[args.offset : args.offset + args.n]
    if not batch:
        print(f"No PNG images found at offset {args.offset} in {images_dir}", file=sys.stderr)
        raise SystemExit(1)
    return batch


def _load_predictions(
    feats: dict[str, float],
    profile: dict[str, Any] | None,
) -> dict[str, dict[str, float | bool | None]]:
    if profile is None:
        return threshold_predictions(feats)
    return predict_with_profile(feats, profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 M4 Competency Analyzer")
    parser.add_argument("--n", type=int, default=10, help="images to process (default 10)")
    parser.add_argument("--offset", type=int, default=0, help="skip first N images (default 0)")
    parser.add_argument(
        "--append", action="store_true", help="append to existing report.json instead of overwriting"
    )
    parser.add_argument("--profile", type=Path, help="path to a persisted calibration profile")
    parser.add_argument("--image", type=Path, help="score a single PNG instead of scanning a directory")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR, help="directory of PNGs to analyze")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "report.json", help="output JSON report path")
    args = parser.parse_args()

    batch = _iter_images(args)
    profile = load_profile(args.profile) if args.profile is not None else None
    total_available = len(batch) if args.image else len(sorted(args.images_dir.glob("*.png")))

    print(f"\n{'=' * 60}")
    print("  NIH ChestX-ray14 — Competency Analyzer")
    source_label = args.image.name if args.image else f"Offset {args.offset} · Batch {len(batch)}"
    print(f"  {source_label} · Total available {total_available}")
    if args.profile is not None:
        print(f"  Detection mode: calibrated profile ({args.profile})")
    else:
        print("  Detection mode: threshold fallback")
    print(f"{'=' * 60}")

    results: list[tuple[str, list[str]]] = []
    json_records: list[dict[str, Any]] = []
    for img_path in batch:
        try:
            feats = extract_features(str(img_path))
        except Exception as e:
            print(f"\n  [SKIP] {img_path.name} — {e}", file=sys.stderr)
            continue

        predictions = _load_predictions(feats, profile)
        record = build_report_record(img_path.name, feats, predictions)
        found = [e["finding"] for e in record["ddx"]]
        print(format_report(record))
        results.append((img_path.name, found))
        json_records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.append and args.out.exists():
        existing = json.loads(args.out.read_text())
        seen = {r["image"] for r in existing}
        merged = existing + [r for r in json_records if r["image"] not in seen]
        args.out.write_text(json.dumps(merged, indent=2))
        print(f"\n  JSON appended → {args.out}  ({len(merged)} total records)")
    else:
        args.out.write_text(json.dumps(json_records, indent=2))
        print(f"\n  JSON saved → {args.out}")

    _write_dashboard()
    print(f"  Dashboard → {RESULTS_DIR / 'dashboard.html'}")
    print(summary_table(results))
    print(f"\n{'─' * 60}")
    print("  Tier legend:")
    print("    Tier 1 ▲ MUST RECOGNIZE  — life-threatening, act immediately")
    print("    Tier 2 ● Should Recognize — common, M4 expected to diagnose")
    print("    Tier 3 ○ Should Know      — important, but lower urgency")
    print(f"{'─' * 60}\n")


def _write_dashboard() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    shutil.copy(DASHBOARD_TMPL, RESULTS_DIR / "dashboard.html")
    shutil.copy(DASHBOARD_TMPL, RESULTS_DIR / "index.html")


if __name__ == "__main__":
    main()
