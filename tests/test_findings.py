"""Pin THRESHOLDS against the shipped seed fixture.

The seed fixture's `findings.<key>.detected` flags must match what
detect_findings() produces from the same metrics. Any threshold edit that
changes detections will break this test loudly — re-run the seeder if the
change was intentional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analyzer.main import THRESHOLDS, detect_findings

SEED = Path(__file__).parent.parent / "fixtures" / "seed_report.json"


def _records() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(SEED.read_text())
    return data


def test_seed_fixture_exists_and_is_nonempty() -> None:
    records = _records()
    assert len(records) >= 5


def test_seed_findings_match_current_thresholds() -> None:
    for record in _records():
        expected = {k for k, v in record["findings"].items() if v["detected"]}
        actual = set(detect_findings(record["metrics"]))
        assert actual == expected, (
            f"{record['image']}: thresholds drifted from fixture. "
            f"actual={sorted(actual)} expected={sorted(expected)}"
        )


def test_consolidation_and_focal_opacity_can_coexist() -> None:
    """Both findings can fire on the same image after the threshold tuning.

    A diffuse airspace process (consolidation) and a focal lesion
    (focal_opacity) are not mutually exclusive clinically, and the tuned
    cutoffs let them overlap on focal_variance ∈ [0.0122, 0.0200). The
    fixture must contain at least one record where both fire so this fact
    stays exercised.
    """
    fired_both = [
        r["image"]
        for r in _records()
        if r["findings"]["consolidation"]["detected"] and r["findings"]["focal_opacity"]["detected"]
    ]
    assert fired_both, (
        "Seed fixture should include at least one record where both "
        "consolidation and focal_opacity fire — they are no longer mutually exclusive."
    )


def test_thresholds_keys_match_findings_module() -> None:
    """Every THRESHOLDS key must have metadata in m4_findings.FINDINGS."""
    from analyzer.m4_findings import FINDINGS

    assert set(THRESHOLDS.keys()) == set(FINDINGS.keys())
