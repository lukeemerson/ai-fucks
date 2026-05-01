from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyzer.features import METRIC_SCHEMA_VERSION
from analyzer.m4_findings import FINDINGS
from analyzer.profile import (
    PROFILE_FORMAT_VERSION,
    PROFILE_TYPE,
    ProfileError,
    load_profile,
    predict_with_profile,
    save_profile,
)


def _dummy_profile() -> dict[str, object]:
    return {
        "profile_type": PROFILE_TYPE,
        "format_version": PROFILE_FORMAT_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "created_at": "2026-01-01T00:00:00+00:00",
        "training": {"dataset_size": 1, "split_sizes": {"train": 1, "val": 1, "test": 1}},
        "findings": {
            key: {
                "feature_names": ["ctr"],
                "scaler_mean": [0.0],
                "scaler_scale": [1.0],
                "coefficients": [1.0],
                "intercept": -0.25,
                "threshold": 0.5,
                "threshold_source": "validation_f1",
                "train_metrics": {},
                "validation_metrics": {},
                "test_metrics": {},
            }
            for key in FINDINGS
        },
    }


def test_profile_round_trip(tmp_path: Path) -> None:
    profile = _dummy_profile()
    out = tmp_path / "profile.json"
    save_profile(profile, out)
    loaded = load_profile(out)
    assert loaded["profile_type"] == PROFILE_TYPE
    pred = predict_with_profile({"ctr": 1.0}, loaded)
    assert set(pred) == set(FINDINGS)
    assert all(0.0 <= pred[key]["probability"] <= 1.0 for key in pred)


def test_profile_rejects_metric_schema_mismatch(tmp_path: Path) -> None:
    profile = _dummy_profile()
    profile["metric_schema_version"] = "old-schema"
    out = tmp_path / "bad_profile.json"
    out.write_text(json.dumps(profile))
    with pytest.raises(ProfileError):
        load_profile(out)
