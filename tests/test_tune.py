"""Unit tests for analyzer/tune.py.

The compound-rule tuner is small enough that we test it with synthetic,
hand-separable data — pneumothorax positives have low peripheral mean+std,
consolidation positives have high haze + high basal + low focal variance.
A correctly wired grid search must recover those splits with F1 ≈ 1.0.

We intentionally do NOT test against real NIH data here — that's the job of
``analyzer.evaluate`` against a real labels CSV.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from analyzer.tune import COMPOUND_RULES, CompoundRule, split_train_test, tune_compound

_METRIC_KEYS = (
    "ctr",
    "ptx_left_mean",
    "ptx_right_mean",
    "ptx_left_std",
    "ptx_right_std",
    "basal_opacity",
    "bilateral_haze",
    "diaphragm_pos",
    "focal_variance",
    "horiz_band",
)
_BASE_METRICS: dict[str, float] = dict.fromkeys(_METRIC_KEYS, 0.5)


def _rec(image: str, **overrides: float) -> dict[str, Any]:
    return {"image": image, "metrics": dict(_BASE_METRICS, **overrides)}


def _rule(name: str) -> CompoundRule:
    for r in COMPOUND_RULES:
        if r.name == name:
            return r
    raise AssertionError(f"missing rule {name}")


def _expected_test_set(names: list[str], test_frac: float) -> set[str]:
    """Reimplement the documented hash bucketing: SHA-1 first byte < threshold → test.

    Kept independent of ``split_train_test`` so swapping SHA-1 for any other
    stable hash (or flipping ``<`` to ``<=``) flips this assertion.
    """
    threshold = int(round(test_frac * 256))
    return {n for n in names if hashlib.sha1(n.encode("utf-8")).digest()[0] < threshold}


def test_split_is_deterministic_and_disjoint() -> None:
    """Hardcoded inputs land in the exact buckets predicted by the SHA-1 scheme."""
    names = [f"img_{i:04d}.png" for i in range(20)]
    records: list[dict[str, Any]] = [{"image": n, "metrics": {}} for n in names]

    # Pinning the expected partition for img_0000 .. img_0019 with test_frac=0.2:
    # threshold = round(0.2 * 256) = 51, so SHA-1(name)[0] < 51 → test bucket.
    # The three filenames below were chosen by computing those hashes once;
    # changing the hash function or the comparison must break this assertion.
    expected_test = {"img_0002.png", "img_0008.png", "img_0014.png"}
    assert expected_test == _expected_test_set(names, 0.2)

    train1, test1 = split_train_test(records, 0.2)
    train2, test2 = split_train_test(records, 0.2)

    train_names1 = {r["image"] for r in train1}
    test_names1 = {r["image"] for r in test1}
    train_names2 = {r["image"] for r in train2}
    test_names2 = {r["image"] for r in test2}

    # Deterministic: same call twice → same partition.
    assert train_names1 == train_names2
    assert test_names1 == test_names2

    # Exact membership match against the hand-computed expectation.
    assert test_names1 == expected_test
    assert train_names1 == set(names) - expected_test

    # And the basic invariants: disjoint, total coverage.
    assert train_names1.isdisjoint(test_names1)
    assert train_names1 | test_names1 == set(names)


def test_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        split_train_test([], 0.0)
    with pytest.raises(ValueError):
        split_train_test([], 1.0)


def test_tune_pneumothorax_recovers_separable_split() -> None:
    """Positives have low peripheral mean+std; negatives don't. F1 must be ~1."""
    records: list[dict[str, Any]] = []
    labels: dict[str, set[str]] = {}
    for i in range(40):
        name = f"ptx_pos_{i:03d}.png"
        records.append(_rec(name, ptx_left_mean=0.12, ptx_left_std=0.03))
        labels[name] = {"Pneumothorax"}
    for i in range(80):
        name = f"neg_{i:03d}.png"
        records.append(
            _rec(
                name,
                ptx_left_mean=0.55,
                ptx_left_std=0.12,
                ptx_right_mean=0.55,
                ptx_right_std=0.12,
            )
        )
        labels[name] = {"No Finding"}

    train, test = split_train_test(records, 0.2)
    res = tune_compound(_rule("pneumothorax"), train, test, labels)

    assert res is not None
    assert res.train_f1 == pytest.approx(1.0, abs=1e-9)
    # Test set is small but still separable, so F1 should be perfect.
    assert res.test_f1 == pytest.approx(1.0, abs=1e-9)


def test_tune_consolidation_recovers_separable_split() -> None:
    """High haze + high basal + low focal_variance must drive F1 to ~1."""
    records: list[dict[str, Any]] = []
    labels: dict[str, set[str]] = {}
    for i in range(40):
        name = f"cons_pos_{i:03d}.png"
        records.append(
            _rec(
                name,
                bilateral_haze=0.62,
                basal_opacity=0.62,
                focal_variance=0.025,
            )
        )
        labels[name] = {"Consolidation"}
    for i in range(80):
        name = f"neg_{i:03d}.png"
        records.append(
            _rec(
                name,
                bilateral_haze=0.30,
                basal_opacity=0.30,
                focal_variance=0.090,
            )
        )
        labels[name] = {"No Finding"}

    train, test = split_train_test(records, 0.2)
    res = tune_compound(_rule("consolidation"), train, test, labels)

    assert res is not None
    assert res.train_f1 == pytest.approx(1.0, abs=1e-9)
    assert res.test_f1 == pytest.approx(1.0, abs=1e-9)


def test_tune_returns_none_when_train_has_no_positives() -> None:
    records = [_rec(f"neg_{i}.png") for i in range(20)]
    labels = {r["image"]: {"No Finding"} for r in records}
    res = tune_compound(_rule("pneumothorax"), records, [], labels)
    assert res is None


def test_tune_returns_none_when_test_has_no_positives() -> None:
    """Engineered filenames so EVERY positive lands in the train bucket.

    With ``test_frac=0.2`` (threshold=51), the names below all have a SHA-1
    first byte ≥ 51 → train. We add only train-bucket positives, plus a few
    test-bucket negatives so the test split isn't empty. Result: train has
    positives, test does not, and ``tune_compound`` must return ``None``
    rather than scoring an undefined F1 on the held-out side.
    """
    # Positives — every one of these hashes into the train bucket.
    pos_train_names = [
        "fixed_0000.png",
        "fixed_0001.png",
        "fixed_0002.png",
        "fixed_0003.png",
        "fixed_0004.png",
        "fixed_0006.png",
        "fixed_0009.png",
        "fixed_0010.png",
        "fixed_0011.png",
        "fixed_0012.png",
    ]
    # Negatives that hash into the test bucket — keeps the test split non-empty.
    neg_test_names = [
        "fixed_0005.png",
        "fixed_0007.png",
        "fixed_0008.png",
        "fixed_0013.png",
    ]

    records: list[dict[str, Any]] = []
    labels: dict[str, set[str]] = {}
    for name in pos_train_names:
        records.append(_rec(name, ptx_left_mean=0.12, ptx_left_std=0.03))
        labels[name] = {"Pneumothorax"}
    for name in neg_test_names:
        records.append(
            _rec(
                name,
                ptx_left_mean=0.55,
                ptx_left_std=0.12,
                ptx_right_mean=0.55,
                ptx_right_std=0.12,
            )
        )
        labels[name] = {"No Finding"}

    train, test = split_train_test(records, 0.2)
    # Sanity-check the engineered split: positives only in train, test non-empty
    # but contains no positives.
    train_pos = sum(1 for r in train if "Pneumothorax" in labels[r["image"]])
    test_pos = sum(1 for r in test if "Pneumothorax" in labels[r["image"]])
    assert train_pos > 0
    assert len(test) > 0
    assert test_pos == 0

    res = tune_compound(_rule("pneumothorax"), train, test, labels)
    assert res is None
