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


def test_consolidation_and_focal_opacity_are_now_separated() -> None:
    """The conservative recalibration makes diffuse and focal calls exclusive.

    Consolidation now requires ``focal_variance < 0.0200`` while
    focal_opacity requires ``focal_variance >= 0.0200``. Keeping that split
    explicit protects the current more-conservative dashboard behavior.
    """
    fired_both = [
        r["image"]
        for r in _records()
        if r["findings"]["consolidation"]["detected"] and r["findings"]["focal_opacity"]["detected"]
    ]
    assert not fired_both, (
        "Consolidation and focal_opacity should be mutually exclusive under "
        "the current threshold split."
    )


def test_thresholds_keys_match_findings_module() -> None:
    """Every THRESHOLDS key must have metadata in m4_findings.FINDINGS."""
    from analyzer.m4_findings import FINDINGS

    assert set(THRESHOLDS.keys()) == set(FINDINGS.keys())
