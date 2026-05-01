"""Integration tests for :class:`harness.adapters.fs.nih_dataset.NIHDataset`.

These tests run against the synthetic 16-row fixture under
``tests/harness/fixtures/nih/`` (see ``NIH_DATASET_SPEC.md`` section 9).
They assert the composition behaviour of the CSV ingestor + image loader
without touching the real on-disk corpus.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from harness.adapters.fs.nih_csv import NIH14_LABELS
from harness.adapters.fs.nih_dataset import NIHDataset, NIHDatasetConfig
from harness.domain.errors import DataError
from harness.domain.types import Sample

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "nih"
FIXTURE_CSV = FIXTURE_ROOT / "Data_Entry_synthetic.csv"
FIXTURE_IMAGES = FIXTURE_ROOT / "images"

PNG_MAGIC = b"\x89PNG"
EXPECTED_ROW_COUNT = 16
EXPECTED_METADATA_KEYS = frozenset(
    {
        "follow_up",
        "patient_age",
        "patient_gender",
        "view_position",
        "width",
        "height",
        "pixel_spacing_x",
        "pixel_spacing_y",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(
    *,
    csv_path: Path = FIXTURE_CSV,
    images_dir: Path = FIXTURE_IMAGES,
    strict_missing_images: bool = True,
    disk_cache_dir: Path | None = None,
) -> NIHDataset:
    config = NIHDatasetConfig(
        csv_path=csv_path,
        images_dir=images_dir,
        image_size=(8, 8),
        cache_size=4,
        disk_cache_dir=disk_cache_dir,
        strict_missing_images=strict_missing_images,
    )
    return NIHDataset(config)


@pytest.fixture
def chdir_tmp(tmp_path: Path) -> Iterator[Path]:
    """Temporarily change CWD into ``tmp_path`` for the duration of the test."""
    prev = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# Construction & basic shape
# ---------------------------------------------------------------------------


def test_construction_succeeds_with_strict_missing_images_true() -> None:
    dataset = _build(strict_missing_images=True)
    assert isinstance(dataset, NIHDataset)


def test_len_equals_fixture_row_count() -> None:
    dataset = _build()
    assert len(dataset) == EXPECTED_ROW_COUNT


def test_sample_ids_are_csv_row_ordered_png_filenames() -> None:
    dataset = _build()
    ids = dataset.sample_ids()
    assert isinstance(ids, tuple)
    assert len(ids) == EXPECTED_ROW_COUNT
    for sid in ids:
        assert isinstance(sid, str)
        assert sid.endswith(".png")
    # CSV row 0 is 00000001_000.png per the synthetic fixture.
    assert ids[0] == "00000001_000.png"


# ---------------------------------------------------------------------------
# Sample lookup
# ---------------------------------------------------------------------------


def test_get_sample_returns_cardiomegaly_only_for_first_row() -> None:
    dataset = _build()
    sample = dataset.get_sample("00000001_000.png")
    assert isinstance(sample, Sample)
    assert sample.sample_id == "00000001_000.png"
    assert sample.patient_id == "1"

    expected = [0] * 14
    expected[NIH14_LABELS.index("Cardiomegaly")] = 1
    assert sample.labels == tuple(expected)


def test_get_sample_returns_hernia_for_hernia_row() -> None:
    dataset = _build()
    sample = dataset.get_sample("00000004_013.png")
    hernia_idx = NIH14_LABELS.index("Hernia")
    assert sample.labels[hernia_idx] == 1
    # Hernia-only row -> only that bit set.
    assert sum(sample.labels) == 1


def test_get_sample_unknown_id_raises_data_error() -> None:
    dataset = _build()
    with pytest.raises(DataError):
        dataset.get_sample("nonexistent.png")


# ---------------------------------------------------------------------------
# Sample.metadata key set (review C2)
# ---------------------------------------------------------------------------


def test_sample_metadata_has_all_eight_spec_required_keys() -> None:
    dataset = _build()
    sample = dataset.get_sample("00000001_000.png")
    assert frozenset(sample.metadata.keys()) == EXPECTED_METADATA_KEYS


def test_sample_metadata_values_match_first_csv_row() -> None:
    dataset = _build()
    sample = dataset.get_sample("00000001_000.png")
    md = sample.metadata
    assert md["follow_up"] == "0"
    assert md["patient_age"] == "57"
    assert md["patient_gender"] == "M"
    assert md["view_position"] == "PA"
    assert md["width"] == "2682"
    assert md["height"] == "2749"
    # Pixel spacings are stringified floats.
    assert float(md["pixel_spacing_x"]) == pytest.approx(0.143)
    assert float(md["pixel_spacing_y"]) == pytest.approx(0.143)


def test_sample_metadata_dimensions_distinct_per_row() -> None:
    dataset = _build()
    a = dataset.get_sample("00000001_000.png").metadata
    b = dataset.get_sample("00000001_001.png").metadata
    assert a["width"] != b["width"] or a["height"] != b["height"]


def test_sample_metadata_values_are_all_strings() -> None:
    dataset = _build()
    sample = dataset.get_sample("00000003_000.png")
    for key, value in sample.metadata.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


# ---------------------------------------------------------------------------
# Image bytes
# ---------------------------------------------------------------------------


def test_get_image_bytes_returns_png_payload() -> None:
    dataset = _build()
    ds = dataset.load()
    ref = ds.samples[0].image_ref
    payload = dataset.get_image_bytes(ref)
    assert isinstance(payload, bytes)
    assert payload.startswith(PNG_MAGIC)


def test_get_image_bytes_unknown_ref_raises_data_error() -> None:
    dataset = _build()
    with pytest.raises(DataError):
        dataset.get_image_bytes("/no/such/path.png")


def test_get_image_bytes_works_after_chdir(chdir_tmp: Path) -> None:
    """Per spec §13.7: image_ref is absolute; CWD changes must not break
    byte resolution. This is the regression for review C1."""
    # Build the dataset *while* CWD is the tmp dir; the config receives
    # absolute fixture paths so this is the realistic case.
    dataset = _build()
    ref = dataset.load().samples[0].image_ref
    # Now move CWD again to a deeper subdir before resolving bytes.
    deeper = chdir_tmp / "nested" / "subdir"
    deeper.mkdir(parents=True, exist_ok=True)
    os.chdir(deeper)
    payload = dataset.get_image_bytes(ref)
    assert payload.startswith(PNG_MAGIC)


def test_get_image_bytes_with_relative_images_dir_works_after_chdir(
    chdir_tmp: Path,
) -> None:
    """Pass a *relative* ``images_dir`` and ``csv_path`` to the config; the
    ``__post_init__`` resolve must capture the absolute path so subsequent
    chdir does not break ``get_image_bytes``."""
    # Reach the fixture dir relative to chdir_tmp so the input is genuinely
    # a relative path that depends on CWD at construction time.
    rel_csv = Path(os.path.relpath(FIXTURE_CSV, chdir_tmp))
    rel_images = Path(os.path.relpath(FIXTURE_IMAGES, chdir_tmp))
    config = NIHDatasetConfig(
        csv_path=rel_csv,
        images_dir=rel_images,
        image_size=(8, 8),
        cache_size=4,
    )
    dataset = NIHDataset(config)
    ref = dataset.load().samples[0].image_ref
    # Move to a different CWD before fetching bytes.
    other = chdir_tmp / "other"
    other.mkdir()
    os.chdir(other)
    payload = dataset.get_image_bytes(ref)
    assert payload.startswith(PNG_MAGIC)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_constructions_produce_identical_sample_order() -> None:
    a = _build()
    b = _build()
    assert a.sample_ids() == b.sample_ids()
    sa = a.get_sample("00000003_000.png")
    sb = b.get_sample("00000003_000.png")
    assert sa.labels == sb.labels
    assert sa.patient_id == sb.patient_id
    assert sa.metadata == sb.metadata


# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------


def test_label_names_returns_canonical_nih14_tuple() -> None:
    dataset = _build()
    assert dataset.label_names == NIH14_LABELS
    assert len(dataset.label_names) == 14


# ---------------------------------------------------------------------------
# strict_missing_images behaviour
# ---------------------------------------------------------------------------


def _augmented_csv_with_missing_row(tmp_path: Path) -> Path:
    """Copy the fixture CSV, append one row that references a missing PNG."""
    src_rows = list(csv.reader(FIXTURE_CSV.open(newline="", encoding="utf-8")))
    dst = tmp_path / "Data_Entry_with_missing.csv"
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in src_rows:
            writer.writerow(row)
        writer.writerow(
            [
                "99999999_000.png",
                "Cardiomegaly",
                "0",
                "999",
                "50",
                "M",
                "PA",
                "2500",
                "2048",
                "0.143",
                "0.143",
            ]
        )
    return dst


def test_strict_missing_images_false_filters_missing_rows(tmp_path: Path) -> None:
    augmented = _augmented_csv_with_missing_row(tmp_path)
    dataset = _build(csv_path=augmented, strict_missing_images=False)
    # The bogus row must be filtered out; surviving count stays at the original 16.
    assert len(dataset) == EXPECTED_ROW_COUNT
    assert "99999999_000.png" not in dataset.sample_ids()


def test_strict_missing_images_true_raises_for_missing_rows(tmp_path: Path) -> None:
    augmented = _augmented_csv_with_missing_row(tmp_path)
    with pytest.raises(DataError):
        _build(csv_path=augmented, strict_missing_images=True)


# ---------------------------------------------------------------------------
# Config resolution & disk_cache_dir creation (review C3)
# ---------------------------------------------------------------------------


def test_config_resolves_relative_images_dir(chdir_tmp: Path) -> None:
    """``__post_init__`` must call ``Path.resolve()`` on inputs."""
    rel_csv = Path(os.path.relpath(FIXTURE_CSV, chdir_tmp))
    rel_images = Path(os.path.relpath(FIXTURE_IMAGES, chdir_tmp))
    config = NIHDatasetConfig(
        csv_path=rel_csv,
        images_dir=rel_images,
        image_size=(8, 8),
        cache_size=1,
    )
    assert config.csv_path.is_absolute()
    assert config.images_dir.is_absolute()
    assert config.csv_path == FIXTURE_CSV.resolve()
    assert config.images_dir == FIXTURE_IMAGES.resolve()


def test_config_creates_disk_cache_dir_if_absent(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "nested"
    assert not cache_dir.exists()
    config = NIHDatasetConfig(
        csv_path=FIXTURE_CSV,
        images_dir=FIXTURE_IMAGES,
        image_size=(8, 8),
        cache_size=1,
        disk_cache_dir=cache_dir,
    )
    assert config.disk_cache_dir is not None
    assert config.disk_cache_dir.is_dir()
    assert config.disk_cache_dir == cache_dir.resolve()


def test_config_disk_cache_dir_is_resolved_absolute(chdir_tmp: Path) -> None:
    rel_cache = Path("rel_cache_dir")
    config = NIHDatasetConfig(
        csv_path=FIXTURE_CSV,
        images_dir=FIXTURE_IMAGES,
        image_size=(8, 8),
        cache_size=1,
        disk_cache_dir=rel_cache,
    )
    assert config.disk_cache_dir is not None
    assert config.disk_cache_dir.is_absolute()
    assert config.disk_cache_dir == (chdir_tmp / "rel_cache_dir").resolve()
