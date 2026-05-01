from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from analyzer.features import METRIC_KEYS, METRIC_SCHEMA_VERSION
from analyzer.m4_findings import FINDINGS

PROFILE_FORMAT_VERSION = 2
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
        importances = model.get("feature_importances")
        if not isinstance(importances, list) or len(importances) != n:
            raise ProfileError(f"{key}: feature_importances must have length {n}")
        if not isinstance(model.get("threshold"), (int, float)):
            raise ProfileError(f"{key}: missing threshold")


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text())
    validate_profile(profile)
    joblib_path = path.with_suffix(".joblib")
    if joblib_path.exists():
        profile["_models"] = joblib.load(joblib_path)
    return profile


def save_profile(
    profile: dict[str, Any],
    path: Path,
    models: dict[str, Any] | None = None,
) -> None:
    serialisable = {k: v for k, v in profile.items() if k != "_models"}
    validate_profile(serialisable)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialisable, indent=2))
    if models is not None:
        joblib.dump(models, path.with_suffix(".joblib"))


def predict_probabilities(feats: dict[str, float], profile: dict[str, Any]) -> dict[str, float]:
    models = profile.get("_models", {})
    probs: dict[str, float] = {}
    for key, model_meta in profile["findings"].items():
        X = [[float(feats[f]) for f in model_meta["feature_names"]]]
        probs[key] = float(models[key].predict_proba(X)[0, 1])
    return probs


def predict_with_profile(
    feats: dict[str, float], profile: dict[str, Any]
) -> dict[str, dict[str, float | bool]]:
    probs = predict_probabilities(feats, profile)
    return {
        key: {
            "probability": prob,
            "detected": prob >= float(profile["findings"][key]["threshold"]),
        }
        for key, prob in probs.items()
    }
