"""Smoke test for :class:`harness.adapters.fs.nih_dataset.NIHDataset`.

Runs against the *real* NIH ChestX-ray14 manifest and a local subset of the
PNG corpus (~5,000 images out of 112,121). Marked ``@pytest.mark.smoke`` so
default ``pytest`` invocations skip it; run explicitly with ``-m smoke``.

Per ``NIH_DATASET_SPEC.md`` section 11 the test asserts only structural
properties; sample-count is non-deterministic across machines.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from harness.adapters.fs.nih_csv import NIH14_LABELS, NIHLabelEncoder
from harness.adapters.fs.nih_dataset import NIHDataset, NIHDatasetConfig

# Resolve the project root from this file's location so the test is portable
# across developer machines (review M2). ``parents[3]`` is
# ``tests/harness/integration/`` -> ``tests/harness/`` -> ``tests/`` -> repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REAL_CSV = PROJECT_ROOT / "Data_Entry_2017_v2020.csv"
REAL_IMAGES = PROJECT_ROOT / "db-test_images" / "images"

PNG_MAGIC = b"\x89PNG"


pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not REAL_CSV.is_file() or not REAL_IMAGES.is_dir(),
        reason=(
            f"local NIH data not available "
            f"(expected {REAL_CSV} and {REAL_IMAGES})"
        ),
    ),
]


def test_smoke_against_real_subset() -> None:
    config = NIHDatasetConfig(
        csv_path=REAL_CSV,
        images_dir=REAL_IMAGES,
        image_size=(64, 64),
        cache_size=128,
        disk_cache_dir=None,
        strict_missing_images=False,
    )
    dataset = NIHDataset(config)

    n = len(dataset)
    # Per spec §11 step 3: tighten to 4,500 <= n <= 5,500 (review M3).
    assert 4_500 <= n <= 5_500, f"expected 4_500 <= n <= 5_500 surviving rows, got {n}"

    ids = dataset.sample_ids()
    assert len(ids) == n
    first_id = ids[0]
    sample = dataset.get_sample(first_id)
    assert sample.sample_id == first_id
    assert isinstance(sample.patient_id, str) and sample.patient_id
    assert len(sample.labels) == 14
    assert isinstance(sample.image_ref, str)

    # Bytes start with PNG magic; do not decode.
    payload = dataset.get_image_bytes(sample.image_ref)
    assert isinstance(payload, bytes)
    assert payload.startswith(PNG_MAGIC)

    # Label-vocabulary sanity.
    ds = dataset.load()
    assert ds.label_names == NIH14_LABELS

    # Label-diversity assertion (spec §11 step 5, review M4):
    # at least 5 distinct NIH labels appear with count >= 1 across the
    # loaded dataset.
    encoder = NIHLabelEncoder()  # only used here for cross-checking len(14)
    assert len(encoder.encode("No Finding")) == 14

    counts: Counter[str] = Counter()
    for s in ds.samples:
        for idx, bit in enumerate(s.labels):
            if bit:
                counts[NIH14_LABELS[idx]] += 1

    distinct_present = {label for label, count in counts.items() if count > 0}
    assert len(distinct_present) >= 5, (
        f"expected >=5 distinct NIH labels with count>=1; got "
        f"{len(distinct_present)} ({sorted(distinct_present)})"
    )

    # Label distribution: rows with >=1 positive vs. all-zero (No Finding).
    positive = 0
    no_finding = 0
    for s in ds.samples:
        if any(s.labels):
            positive += 1
        else:
            no_finding += 1

    # Informational; pytest -s surfaces the line.
    print(
        f"\n[smoke] dataset_size={n} positive={positive} "
        f"no_finding={no_finding} distinct_labels={len(distinct_present)}"
    )
