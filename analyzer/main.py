from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from analyzer.features import extract_features
from analyzer.m4_findings import FINDINGS, tier_badge

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "db-test_images" / "images"
DASHBOARD_TMPL = Path(__file__).parent / "dashboard.html"

# Each lambda receives the feature dict from extract_features() and returns bool.
# Cutoffs are hand-tuned against NIH ChestX-ray14 pixel statistics (grayscale 0-1).
# Consolidation requires diffuse airspace signal (haze + basal opacity) AND
# low-variance fields. Focal_opacity flags a high-variance focal hot-spot. Both
# can fire on the same image when a diffuse process coexists with a focal lesion
# — which is clinically realistic.
THRESHOLDS: dict[str, Callable[[dict[str, float]], bool]] = {
    "cardiomegaly": lambda f: f["ctr"] > 0.2217,
    "pneumothorax": lambda f: (
        (f["ptx_left_mean"] < 0.33 and f["ptx_left_std"] < 0.14)
        or (f["ptx_right_mean"] < 0.33 and f["ptx_right_std"] < 0.14)
    ),
    "pleural_effusion": lambda f: f["basal_opacity"] > 0.52,
    "pulmonary_edema": lambda f: f["bilateral_haze"] > 0.3533,
    "consolidation": lambda f: (
        f["bilateral_haze"] > 0.3500 and f["basal_opacity"] > 0.4500 and f["focal_variance"] < 0.0200
    ),
    "atelectasis": lambda f: f["horiz_band"] > 0.065,
    "emphysema": lambda f: f["diaphragm_pos"] > 0.6670,
    "focal_opacity": lambda f: f["focal_variance"] >= 0.0122,
}


def detect_findings(feats: dict[str, float]) -> list[str]:
    return [key for key, test in THRESHOLDS.items() if test(feats)]


def format_report(image_name: str, feats: dict[str, float], found: list[str]) -> str:
    lines = [f"\n{'─' * 60}", f"  Image: {image_name}", f"{'─' * 60}"]

    if not found:
        lines.append("  No significant findings detected.")
        lines.append("  M4 note: Normal chest X-ray — confirm bilateral lung fields,")
        lines.append("           clear costophrenic angles, normal cardiac silhouette.")
        return "\n".join(lines)

    for key in sorted(found, key=lambda k: FINDINGS[k].tier):
        f = FINDINGS[key]
        lines.append(f"\n  {tier_badge(f.tier)} {f.name}")
        lines.append(f"    Finding : {f.description}")
        lines.append(f"    M4 action: {f.m4_action}")

    lines.append("\n  Raw metrics:")
    lines.append(
        f"    CTR={feats['ctr']:.2f}  Basal={feats['basal_opacity']:.2f}  "
        f"Haze={feats['bilateral_haze']:.2f}  FocalVar={feats['focal_variance']:.3f}  "
        f"Diaphragm={feats['diaphragm_pos']:.2f}"
    )
    return "\n".join(lines)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 M4 Competency Analyzer")
    parser.add_argument("--n", type=int, default=10, help="images to process (default 10)")
    parser.add_argument("--offset", type=int, default=0, help="skip first N images (default 0)")
    parser.add_argument(
        "--append", action="store_true", help="append to existing report.json instead of overwriting"
    )
    args = parser.parse_args()

    all_images = sorted(IMAGES_DIR.glob("*.png"))
    batch = all_images[args.offset : args.offset + args.n]
    if not batch:
        print(f"No PNG images found at offset {args.offset} in {IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  NIH ChestX-ray14 — Competency Analyzer")
    print(f"  Offset {args.offset} · Batch {len(batch)} · Total available {len(all_images)}")
    print(f"{'=' * 60}")

    results: list[tuple[str, list[str]]] = []
    json_records: list[dict[str, Any]] = []
    for img_path in batch:
        try:
            feats = extract_features(str(img_path))
        except Exception as e:
            print(f"\n  [SKIP] {img_path.name} — {e}", file=sys.stderr)
            continue
        found = detect_findings(feats)
        print(format_report(img_path.name, feats, found))
        results.append((img_path.name, found))
        json_records.append(
            {
                "image": img_path.name,
                "findings": {
                    key: {
                        "detected": key in found,
                        "tier": FINDINGS[key].tier,
                        "tier_label": FINDINGS[key].tier_label,
                        "name": FINDINGS[key].name,
                        "description": FINDINGS[key].description,
                        "m4_action": FINDINGS[key].m4_action,
                    }
                    for key in THRESHOLDS
                },
                "metrics": {k: round(v, 4) for k, v in feats.items()},
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "report.json"

    if args.append and out_path.exists():
        existing = json.loads(out_path.read_text())
        seen = {r["image"] for r in existing}
        merged = existing + [r for r in json_records if r["image"] not in seen]
        out_path.write_text(json.dumps(merged, indent=2))
        print(f"\n  JSON appended → {out_path}  ({len(merged)} total records)")
    else:
        out_path.write_text(json.dumps(json_records, indent=2))
        print(f"\n  JSON saved → {out_path}")

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
    # The HTML template is the single source of truth in analyzer/.
    # results/ is gitignored and rebuilt on each run / first server boot.
    shutil.copy(DASHBOARD_TMPL, RESULTS_DIR / "dashboard.html")
    shutil.copy(DASHBOARD_TMPL, RESULTS_DIR / "index.html")


if __name__ == "__main__":
    main()
