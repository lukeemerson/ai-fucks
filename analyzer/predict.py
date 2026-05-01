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

# Static DDx considerations per finding. Consolidation is handled dynamically
# using cross-finding probabilities — see _consolidation_considerations().
CONSIDERATIONS: dict[str, list[str]] = {
    "cardiomegaly": [
        "Dilated cardiomyopathy / decompensated CHF",
        "Valvular disease (MR, AR, MS)",
        "Pericardial effusion",
        "AP projection artifact — confirm on PA film",
    ],
    "pneumothorax": [
        "Tension PTX — emergency if tracheal deviation / haemodynamic instability",
        "Simple PTX — size guides management",
        "Apical bleb / bulla",
        "Skin fold artifact — clinical correlation required",
    ],
    "pleural_effusion": [
        "Parapneumonic / empyema",
        "Transudate: CHF, cirrhosis, nephrotic syndrome (Light's criteria)",
        "Malignant effusion",
        "Haemothorax",
    ],
    "pulmonary_edema": [
        "Cardiogenic: decompensated CHF — check BNP, Echo",
        "Non-cardiogenic / ARDS — sepsis, aspiration, transfusion reaction",
        "Fluid overload",
    ],
    "atelectasis": [
        "Mucus plugging — incentive spirometry, chest PT",
        "Extrinsic compression / adjacent effusion",
        "Postoperative / splinting",
        "Endobronchial lesion if lobar — consider bronchoscopy",
    ],
    "emphysema": [
        "COPD / smoking-related",
        "Alpha-1 antitrypsin deficiency (consider in younger patients)",
        "Asthma — hyperinflation during acute exacerbation",
    ],
    "focal_opacity": [
        "Primary lung malignancy — CT chest, PET if > 8 mm (Fleischner guidelines)",
        "Metastasis",
        "Infectious granuloma (TB, histoplasma, coccidioides)",
        "Hamartoma / benign lesion",
        "Focal pneumonia / aspiration",
    ],
}


def _consolidation_considerations(all_probs: dict[str, float | None]) -> list[str]:
    """Rank consolidation DDx using cross-finding probabilities."""
    consids = ["Lobar pneumonia / CAP — culture, start empiric antibiotics"]
    if (all_probs.get("pulmonary_edema") or 0.0) >= 0.40:
        consids.append("Pulmonary edema pattern (bilateral haze elevated — see above)")
    if (all_probs.get("pleural_effusion") or 0.0) >= 0.40:
        consids.append("Parapneumonic effusion (see pleural effusion above)")
    if (all_probs.get("atelectasis") or 0.0) >= 0.40:
        consids.append("Atelectasis / volume loss (see above)")
    consids.append("Cannot exclude aspiration pneumonitis")
    consids.append("Post-obstructive if lobar without fever — consider bronchoscopy")
    return consids


def _confidence_label(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    if probability >= 0.55:
        return "medium"
    return "low"


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
    all_probs: dict[str, float | None] = {k: v.get("probability") for k, v in predictions.items()}
    ddx: list[dict[str, Any]] = []
    for key, finding in FINDINGS.items():
        pred = predictions.get(key, {})
        if not pred.get("detected"):
            continue
        prob = pred.get("probability")
        ddx.append({
            "finding": key,
            "name": finding.name,
            "tier": finding.tier,
            "tier_label": finding.tier_label,
            "probability": prob,
            "confidence": _confidence_label(float(prob)) if prob is not None else None,
            "description": finding.description,
            "m4_action": finding.m4_action,
            "considerations": (
                _consolidation_considerations(all_probs)
                if key == "consolidation"
                else CONSIDERATIONS.get(key, [])
            ),
        })
    ddx.sort(key=lambda e: (e["tier"], -(e["probability"] or 0.0)))
    return {"image": image_name, "metrics": {k: round(v, 4) for k, v in feats.items()}, "ddx": ddx}


def format_report(record: dict[str, Any]) -> str:
    image_name = record["image"]
    feats = record["metrics"]
    ddx = record.get("ddx", [])
    lines = [f"\n{'─' * 60}", f"  Image: {image_name}", f"{'─' * 60}"]

    if not ddx:
        lines.append("  No significant findings detected.")
        lines.append("  M4 note: Normal chest X-ray — confirm bilateral lung fields,")
        lines.append("           clear costophrenic angles, normal cardiac silhouette.")
        return "\n".join(lines)

    for entry in ddx:
        conf = entry.get("confidence")
        prob = entry.get("probability")
        conf_str = f" — {conf.upper()}" if conf else ""
        prob_str = f" ({float(prob):.0%})" if prob is not None else ""
        lines.append(f"\n  {tier_badge(entry['tier'])} {entry['name']}{conf_str}{prob_str}")
        lines.append(f"    Finding : {entry['description']}")
        if entry.get("considerations"):
            lines.append("    DDx:")
            for i, c in enumerate(entry["considerations"], 1):
                lines.append(f"      {i}. {c}")
        lines.append(f"    M4 action: {entry['m4_action']}")

    lines.append("\n  Raw metrics:")
    lines.append(
        f"    CTR={feats['ctr']:.2f}  Basal={feats['basal_opacity']:.2f}  "
        f"Haze={feats['bilateral_haze']:.2f}  FocalVar={feats['focal_variance']:.3f}  "
        f"Diaphragm={feats['diaphragm_pos']:.2f}"
    )
    return "\n".join(lines)
