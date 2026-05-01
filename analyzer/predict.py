from __future__ import annotations

from collections.abc import Callable
from typing import Any

from analyzer.m4_findings import FINDINGS, tier_badge

ThresholdRules = dict[str, Callable[[dict[str, float]], bool]]

# Conservative fallback rules for environments without a trained profile.
DEFAULT_THRESHOLD_RULES: ThresholdRules = {
    "cardiomegaly": lambda f: f["ctr"] > 0.3000,
    "pneumothorax": lambda f: (
        (f["ptx_left_mean"] < 0.42 and f["ptx_left_std"] < 0.12)
        or (f["ptx_right_mean"] < 0.42 and f["ptx_right_std"] < 0.12)
    ),
    "pleural_effusion": lambda f: f["basal_opacity"] > 0.7800,
    "pulmonary_edema": lambda f: f["bilateral_haze"] > 0.6200,
    "consolidation": lambda f: (
        f["bilateral_haze"] > 0.6200 and f["basal_opacity"] > 0.8000 and f["focal_variance"] < 0.0200
    ),
    "atelectasis": lambda f: f["horiz_band"] > 0.0800,
    "emphysema": lambda f: f["diaphragm_pos"] > 0.8600,
    "focal_opacity": lambda f: f["focal_variance"] >= 0.0200,
}


def detect_findings(feats: dict[str, float], rules: ThresholdRules | None = None) -> list[str]:
    active_rules = DEFAULT_THRESHOLD_RULES if rules is None else rules
    return [key for key, test in active_rules.items() if test(feats)]


def threshold_predictions(
    feats: dict[str, float],
    rules: ThresholdRules | None = None,
) -> dict[str, dict[str, bool | None]]:
    active_rules = DEFAULT_THRESHOLD_RULES if rules is None else rules
    return {
        key: {"detected": bool(test(feats)), "probability": None}
        for key, test in active_rules.items()
    }


def build_report_record(
    image_name: str,
    feats: dict[str, float],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "image": image_name,
        "findings": {
            key: {
                "detected": bool(predictions[key]["detected"]),
                "tier": FINDINGS[key].tier,
                "tier_label": FINDINGS[key].tier_label,
                "name": FINDINGS[key].name,
                "description": FINDINGS[key].description,
                "m4_action": FINDINGS[key].m4_action,
                "probability": predictions[key]["probability"],
            }
            for key in FINDINGS
        },
        "metrics": {k: round(v, 4) for k, v in feats.items()},
    }


def format_report(
    image_name: str,
    feats: dict[str, float],
    predictions: dict[str, dict[str, Any]],
) -> str:
    found = [key for key, pred in predictions.items() if pred["detected"]]
    lines = [f"\n{'─' * 60}", f"  Image: {image_name}", f"{'─' * 60}"]

    if not found:
        lines.append("  No significant findings detected.")
        lines.append("  M4 note: Normal chest X-ray — confirm bilateral lung fields,")
        lines.append("           clear costophrenic angles, normal cardiac silhouette.")
        return "\n".join(lines)

    for key in sorted(found, key=lambda k: FINDINGS[k].tier):
        finding = FINDINGS[key]
        lines.append(f"\n  {tier_badge(finding.tier)} {finding.name}")
        lines.append(f"    Finding : {finding.description}")
        lines.append(f"    M4 action: {finding.m4_action}")
        prob = predictions[key].get("probability")
        if prob is not None:
            lines.append(f"    Confidence: {float(prob):.2%}")

    lines.append("\n  Raw metrics:")
    lines.append(
        f"    CTR={feats['ctr']:.2f}  Basal={feats['basal_opacity']:.2f}  "
        f"Haze={feats['bilateral_haze']:.2f}  FocalVar={feats['focal_variance']:.3f}  "
        f"Diaphragm={feats['diaphragm_pos']:.2f}"
    )
    return "\n".join(lines)
