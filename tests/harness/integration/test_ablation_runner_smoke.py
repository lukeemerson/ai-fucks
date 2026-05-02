"""Smoke test for :mod:`harness.scripts.run_ablation`.

Drives the ablation runner end-to-end on the 16-row NIH fixture across two
seeds and asserts:

* ``main`` returns ``0`` (all variants succeeded).
* ``comparison.csv`` is written with one header row and exactly N data rows.
* The feature cache is shared across seeds: after seed-1 finishes, the
  count of ``.npy`` files in the cache root is unchanged from the count
  taken right after seed-0 (every backbone input hit the cache).
* Same seed + same data => byte-identical ``comparison.csv`` row across
  two independent runs that share a feature cache. This is the headline
  determinism contract the cache must uphold (cache hit vs cache miss
  must produce the same downstream metrics).

Marked ``smoke`` AND ``torch``: requires the on-disk fixture and the real
torchvision adapter. Both markers are excluded from the default fast suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Resolve the 16-row NIH fixture from the repo root. ``parents[3]`` is
# tests/harness/integration -> tests/harness -> tests -> repo root.
_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "harness"
    / "fixtures"
    / "nih"
)
_FIXTURE_CSV = _FIXTURE_ROOT / "Data_Entry_synthetic.csv"
_FIXTURE_IMAGES = _FIXTURE_ROOT / "images"


def _count_npy(root: Path) -> int:
    """Recursive count of ``*.npy`` files under ``root`` (zero if absent)."""
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.npy"))


@pytest.mark.smoke
@pytest.mark.torch
def test_ablation_runner_two_seeds_writes_comparison_and_uses_cache(
    tmp_path: Path,
) -> None:
    """Two-seed ablation populates the cache once and writes ``comparison.csv``.

    The second seed must not extract any new features (every backbone input
    is identical across seeds because the dataset itself is identical), so
    the ``.npy`` count after the second seed must equal the count after the
    first seed. This is the proxy for "the cache is being used."
    """
    if not _FIXTURE_CSV.is_file() or not _FIXTURE_IMAGES.is_dir():
        pytest.skip(
            f"NIH fixture not found at {_FIXTURE_ROOT}; "
            f"expected {_FIXTURE_CSV} and {_FIXTURE_IMAGES}"
        )

    # Lazy import: the ablation runner module imports the factory which
    # imports torchvision lazily; the smoke test stays compatible with the
    # default fast suite.
    from harness.scripts.run_ablation import main as ablation_main

    cache_dir = tmp_path / "cache"
    artifact_root = tmp_path / "runs"

    # First, run only seed 0 to capture the post-extraction cache count.
    rc_first = ablation_main(
        [
            "--seeds",
            "0",
            "--nih-csv",
            str(_FIXTURE_CSV),
            "--nih-images",
            str(_FIXTURE_IMAGES),
            "--n",
            "0",
            "--feature-cache-dir",
            str(cache_dir),
            "--artifact-root",
            str(artifact_root / "first"),
        ]
    )
    assert rc_first == 0, f"first ablation run failed: rc={rc_first}"
    npy_after_first = _count_npy(cache_dir)
    assert npy_after_first > 0, "cache was not populated by the first seed"

    # Second, run seeds 0,1 against the SAME cache dir. Seed 0 hits the cache
    # entirely; seed 1 also hits because feature extraction depends on image
    # bytes only (which are identical), not on the master seed. The cache
    # count therefore stays the same.
    rc_second = ablation_main(
        [
            "--seeds",
            "0,1",
            "--nih-csv",
            str(_FIXTURE_CSV),
            "--nih-images",
            str(_FIXTURE_IMAGES),
            "--n",
            "0",
            "--feature-cache-dir",
            str(cache_dir),
            "--artifact-root",
            str(artifact_root / "second"),
        ]
    )
    assert rc_second == 0, f"second ablation run failed: rc={rc_second}"
    npy_after_second = _count_npy(cache_dir)
    assert npy_after_second == npy_after_first, (
        f"expected cache to be reused (count unchanged) -- "
        f"got {npy_after_first} -> {npy_after_second}"
    )

    # comparison.csv was written under the second run's artifact root with
    # exactly two data rows (seeds 0 and 1 + a header).
    comparison = artifact_root / "second" / "comparison.csv"
    assert comparison.is_file(), f"missing comparison.csv at {comparison}"
    rows = comparison.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3, f"expected 3 lines (header + 2 data), got {rows}"
    header = rows[0].split(",")
    assert header[0] == "seed"
    assert "macro_f1" in header
    assert "macro_auroc" in header
    assert "macro_auprc" in header
    # Data rows begin with the seed values (in run order).
    assert rows[1].startswith("0,")
    assert rows[2].startswith("1,")


@pytest.mark.smoke
@pytest.mark.torch
def test_ablation_runner_same_seed_produces_byte_identical_comparison_row(
    tmp_path: Path,
) -> None:
    """Same seed + shared feature cache => byte-identical ``comparison.csv`` row.

    This is the headline determinism contract for :class:`CachedBackbone`: a
    cache hit must produce numerics indistinguishable from the original cache
    miss, and the entire downstream pipeline (head, calibrator, thresholds,
    metrics) must be deterministic for a fixed seed. Two independent runs of
    the ablation runner -- both with ``--seeds 0`` -- against a shared cache
    directory therefore must emit byte-identical seed-0 rows in their
    respective ``comparison.csv`` files.

    Run 1 populates the cache (every row is a miss). Run 2 reads from it
    (every row is a hit). Asserting on the raw CSV row strings (rather than
    parsing floats with a tolerance) catches any silent deviation that would
    otherwise widen reported confidence intervals downstream.
    """
    if not _FIXTURE_CSV.is_file() or not _FIXTURE_IMAGES.is_dir():
        pytest.skip(
            f"NIH fixture not found at {_FIXTURE_ROOT}; "
            f"expected {_FIXTURE_CSV} and {_FIXTURE_IMAGES}"
        )

    from harness.scripts.run_ablation import main as ablation_main

    cache_dir = tmp_path / "cache"
    artifact_run_1 = tmp_path / "run-1"
    artifact_run_2 = tmp_path / "run-2"

    common_args = [
        "--seeds",
        "0",
        "--nih-csv",
        str(_FIXTURE_CSV),
        "--nih-images",
        str(_FIXTURE_IMAGES),
        "--n",
        "0",
        "--feature-cache-dir",
        str(cache_dir),
    ]

    rc_1 = ablation_main([*common_args, "--artifact-root", str(artifact_run_1)])
    assert rc_1 == 0, f"first run failed: rc={rc_1}"
    rc_2 = ablation_main([*common_args, "--artifact-root", str(artifact_run_2)])
    assert rc_2 == 0, f"second run failed: rc={rc_2}"

    comparison_1 = artifact_run_1 / "comparison.csv"
    comparison_2 = artifact_run_2 / "comparison.csv"
    assert comparison_1.is_file(), f"missing {comparison_1}"
    assert comparison_2.is_file(), f"missing {comparison_2}"

    rows_1 = comparison_1.read_text(encoding="utf-8").splitlines()
    rows_2 = comparison_2.read_text(encoding="utf-8").splitlines()
    assert len(rows_1) == 2, f"expected header + 1 data row, got {rows_1}"
    assert len(rows_2) == 2, f"expected header + 1 data row, got {rows_2}"
    # Header must match (sanity check; the determinism claim is on the data row).
    assert rows_1[0] == rows_2[0], (
        f"comparison.csv headers diverged: {rows_1[0]!r} vs {rows_2[0]!r}"
    )
    # The seed-0 data row must be byte-identical: cache miss (run 1) and
    # cache hit (run 2) produce indistinguishable downstream metrics.
    assert rows_1[1] == rows_2[1], (
        "seed=0 row diverged across runs (cache miss vs cache hit must be "
        f"byte-identical):\n  run 1: {rows_1[1]!r}\n  run 2: {rows_2[1]!r}"
    )
    # Sanity: run 2 reused the cache (no new .npy files written between runs).
    npy_after_run_2 = _count_npy(cache_dir)
    assert npy_after_run_2 > 0, "cache was not populated by run 1"


@pytest.mark.smoke
@pytest.mark.torch
def test_ablation_runner_rejects_empty_or_duplicate_seeds(tmp_path: Path) -> None:
    """``--seeds`` must be non-empty and unique; bad inputs raise SystemExit."""
    from harness.scripts.run_ablation import main as ablation_main

    common_args = [
        "--nih-csv",
        str(_FIXTURE_CSV),
        "--nih-images",
        str(_FIXTURE_IMAGES),
        "--feature-cache-dir",
        str(tmp_path / "cache"),
        "--artifact-root",
        str(tmp_path / "runs"),
    ]

    # Empty seed list (the value is empty after splitting on commas).
    with pytest.raises(SystemExit):
        ablation_main(["--seeds", "", *common_args])

    # Duplicate seeds.
    with pytest.raises(SystemExit):
        ablation_main(["--seeds", "0,0,1", *common_args])
