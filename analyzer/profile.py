from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from analyzer.features import METRIC_KEYS, METRIC_SCHEMA_VERSION
from analyzer.m4_findings import FINDINGS

PROFILE_FORMAT_VERSION = 1
PROFILE_TYPE = "cxr-calibration-profile"


class ProfileError(ValueError):
    pass


def validate_profile(profile: dict[str, Any]) -> None:
    if profile.get("profile_type") != PROFILE_TYPE:
        raise ProfileError(f"unsupported profile type: {profile.get('profile_type')!r}")
    if profile.get("format_version") != PROFILE_FORMAT_VERSION:
        raise ProfileError(f"unsupported format version: {profile.get('format_version')!r}")
    if profile.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
        raise ProfileError(
            "profile metric schema version does not match the current feature extractor "
            f"({profile.get('metric_schema_version')!r} != {METRIC_SCHEMA_VERSION!r})"
        )

    findings = profile.get("findings")
    if not isinstance(findings, dict) or set(findings) != set(FINDINGS):
        raise ProfileError("profile findings do not match the known finding set")

    known_metrics = set(METRIC_KEYS)
    for key, model in findings.items():
        feature_names = model.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise ProfileError(f"{key}: missing feature_names")
        if not set(feature_names).issubset(known_metrics):
            raise ProfileError(f"{key}: unknown feature in profile")
        n = len(feature_names)
        for field in ("scaler_mean", "scaler_scale", "coefficients"):
            values = model.get(field)
            if not isinstance(values, list) or len(values) != n:
                raise ProfileError(f"{key}: {field} must have length {n}")
        if not isinstance(model.get("intercept"), (int, float)):
            raise ProfileError(f"{key}: missing intercept")
        if not isinstance(model.get("threshold"), (int, float)):
            raise ProfileError(f"{key}: missing threshold")


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text())
    validate_profile(profile)
    return profile


def save_profile(profile: dict[str, Any], path: Path) -> None:
    validate_profile(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def predict_probabilities(feats: dict[str, float], profile: dict[str, Any]) -> dict[str, float]:
    probs: dict[str, float] = {}
    for key, model in profile["findings"].items():
        z = float(model["intercept"])
        for feature_name, mean, scale, coef in zip(
            model["feature_names"],
            model["scaler_mean"],
            model["scaler_scale"],
            model["coefficients"],
            strict=True,
        ):
            denom = float(scale) if float(scale) else 1.0
            z += ((float(feats[feature_name]) - float(mean)) / denom) * float(coef)
        probs[key] = _sigmoid(z)
    return probs


def predict_with_profile(feats: dict[str, float], profile: dict[str, Any]) -> dict[str, dict[str, float | bool]]:
    probs = predict_probabilities(feats, profile)
    return {
        key: {
            "probability": prob,
            "detected": prob >= float(profile["findings"][key]["threshold"]),
        }
        for key, prob in probs.items()
    }
