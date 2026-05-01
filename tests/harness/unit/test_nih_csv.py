"""Unit tests for the NIH CSV ingestor and label encoder.

Per spec ``harness/adapters/fs/docs/NIH_DATASET_SPEC.md`` sections 2, 3, 4.1,
7, 12 (Wave 2a) and 13. Pure CSV / label-encoding tests; no image I/O.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from harness.adapters.fs.nih_csv import (
    NIH14_LABELS,
    NIHCsvIngestor,
    NIHLabelEncoder,
    NIHRecord,
)
from harness.domain.errors import DataError

FIXTURE_CSV = (
    Path(__file__).parent.parent / "fixtures" / "nih" / "Data_Entry_synthetic.csv"
)


# ---------------------------------------------------------------------------
# NIH14_LABELS module-level constant
# ---------------------------------------------------------------------------


def test_nih14_labels_has_14_canonical_entries() -> None:
    assert len(NIH14_LABELS) == 14
    assert NIH14_LABELS[0] == "Atelectasis"
    assert NIH14_LABELS[1] == "Cardiomegaly"
    assert NIH14_LABELS[2] == "Effusion"
    assert NIH14_LABELS[3] == "Infiltration"
    assert NIH14_LABELS[4] == "Mass"
    assert NIH14_LABELS[5] == "Nodule"
    assert NIH14_LABELS[6] == "Pneumonia"
    assert NIH14_LABELS[7] == "Pneumothorax"
    assert NIH14_LABELS[8] == "Consolidation"
    assert NIH14_LABELS[9] == "Edema"
    assert NIH14_LABELS[10] == "Emphysema"
    assert NIH14_LABELS[11] == "Fibrosis"
    assert NIH14_LABELS[12] == "Pleural_Thickening"
    assert NIH14_LABELS[13] == "Hernia"


# ---------------------------------------------------------------------------
# NIHLabelEncoder
# ---------------------------------------------------------------------------


def test_encode_single_label_cardiomegaly() -> None:
    encoder = NIHLabelEncoder()
    vec = encoder.encode("Cardiomegaly")
    assert len(vec) == 14
    assert vec[1] == 1
    assert sum(vec) == 1


def test_encode_single_label_atelectasis_index_zero() -> None:
    encoder = NIHLabelEncoder()
    vec = encoder.encode("Atelectasis")
    assert vec[0] == 1
    assert sum(vec) == 1


def test_encode_single_label_hernia_index_thirteen() -> None:
    encoder = NIHLabelEncoder()
    vec = encoder.encode("Hernia")
    assert vec[13] == 1
    assert sum(vec) == 1


def test_encode_multi_label_three_indices_set() -> None:
    encoder = NIHLabelEncoder()
    vec = encoder.encode("Mass|Nodule|Pneumonia")
    assert len(vec) == 14
    assert vec[4] == 1  # Mass
    assert vec[5] == 1  # Nodule
    assert vec[6] == 1  # Pneumonia
    assert sum(vec) == 3


def test_encode_multi_label_order_independent() -> None:
    encoder = NIHLabelEncoder()
    a = encoder.encode("Cardiomegaly|Effusion")
    b = encoder.encode("Effusion|Cardiomegaly")
    assert a == b
    assert a[1] == 1
    assert a[2] == 1
    assert sum(a) == 2


def test_encode_no_finding_all_zeros() -> None:
    encoder = NIHLabelEncoder()
    vec = encoder.encode("No Finding")
    assert vec == (0,) * 14


def test_encode_empty_string_raises_data_error() -> None:
    encoder = NIHLabelEncoder()
    with pytest.raises(DataError, match=r"empty"):
        encoder.encode("")


def test_encode_none_raises_data_error() -> None:
    """The encoder accepts ``str | None`` so callers may forward an absent
    cell verbatim; ``None`` normalises to the same error path as empty."""
    encoder = NIHLabelEncoder()
    with pytest.raises(DataError, match=r"empty"):
        encoder.encode(None)


def test_encode_whitespace_padding_is_tolerated() -> None:
    """Spec 12 / Wave 2a: defensive whitespace stripping is required."""
    encoder = NIHLabelEncoder()
    vec = encoder.encode(" Cardiomegaly | Effusion ")
    assert vec[1] == 1
    assert vec[2] == 1
    assert sum(vec) == 2


def test_encode_unknown_label_raises_data_error_with_label_name() -> None:
    encoder = NIHLabelEncoder()
    with pytest.raises(DataError, match=r"AlienFinding"):
        encoder.encode("AlienFinding")


def test_encode_duplicate_label_in_same_row_is_idempotent() -> None:
    """Defensive: 'Mass|Mass' encodes to the same vector as 'Mass'."""
    encoder = NIHLabelEncoder()
    vec = encoder.encode("Mass|Mass")
    assert vec[4] == 1
    assert sum(vec) == 1


def test_encode_is_deterministic() -> None:
    encoder = NIHLabelEncoder()
    a = encoder.encode("Cardiomegaly|Effusion|Hernia")
    b = encoder.encode("Cardiomegaly|Effusion|Hernia")
    assert a == b
    assert tuple(a) == tuple(b)


def test_encode_returns_tuple_of_ints() -> None:
    encoder = NIHLabelEncoder()
    vec = encoder.encode("Cardiomegaly")
    assert isinstance(vec, tuple)
    for v in vec:
        assert isinstance(v, int)
        assert v in (0, 1)


# ---------------------------------------------------------------------------
# NIHCsvIngestor
# ---------------------------------------------------------------------------


def test_ingest_synthetic_returns_16_records() -> None:
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    assert isinstance(records, tuple)
    assert len(records) == 16
    for r in records:
        assert isinstance(r, NIHRecord)


def test_ingest_first_record_fields() -> None:
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    first = records[0]
    assert first.image_index == "00000001_000.png"
    assert first.patient_id == "1"
    assert first.follow_up == 0
    assert first.age == 57
    assert first.gender == "M"
    assert first.view_position == "PA"
    # Cardiomegaly only
    assert first.label_vector[1] == 1
    assert sum(first.label_vector) == 1
    # Spec §7: width/height/pixel_spacing are persisted on the record.
    assert first.width == 2682
    assert first.height == 2749
    assert first.pixel_spacing_x == pytest.approx(0.143)
    assert first.pixel_spacing_y == pytest.approx(0.143)


def test_ingest_record_dimensions_distinct_per_row() -> None:
    """The fixture rows have distinct widths/heights; verify positional parse."""
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    by_id = {r.image_index: r for r in records}
    # 00000001_001.png,...,2894,2729,...
    r = by_id["00000001_001.png"]
    assert r.width == 2894
    assert r.height == 2729
    # 00000003_000.png,...,2700,2700,...
    r = by_id["00000003_000.png"]
    assert r.width == 2700
    assert r.height == 2700


def test_ingest_record_pixel_spacings_are_floats() -> None:
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    for r in records:
        assert isinstance(r.pixel_spacing_x, float)
        assert isinstance(r.pixel_spacing_y, float)
        assert r.pixel_spacing_x > 0.0
        assert r.pixel_spacing_y > 0.0


def test_ingest_no_finding_row_is_all_zeros() -> None:
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    by_id = {r.image_index: r for r in records}
    nf = by_id["00000002_001.png"]
    assert nf.label_vector == (0,) * 14


def test_ingest_multi_label_row_sets_three_bits() -> None:
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    by_id = {r.image_index: r for r in records}
    multi = by_id["00000003_000.png"]
    # Mass|Nodule|Pneumonia -> indices 4, 5, 6
    assert multi.label_vector[4] == 1
    assert multi.label_vector[5] == 1
    assert multi.label_vector[6] == 1
    assert sum(multi.label_vector) == 3


def test_ingest_preserves_csv_row_order() -> None:
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    expected_order = [
        "00000001_000.png",
        "00000001_001.png",
        "00000001_002.png",
        "00000002_000.png",
        "00000002_001.png",
        "00000003_000.png",
        "00000003_001.png",
        "00000003_002.png",
        "00000003_003.png",
        "00000004_010.png",
        "00000004_011.png",
        "00000004_012.png",
        "00000004_013.png",
        "00000004_014.png",
        "00000005_000.png",
        "00000005_001.png",
    ]
    assert [r.image_index for r in records] == expected_order


def test_ingest_hernia_row_sets_index_thirteen() -> None:
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    by_id = {r.image_index: r for r in records}
    h = by_id["00000004_013.png"]
    assert h.label_vector[13] == 1
    assert sum(h.label_vector) == 1


def test_ingest_duplicate_image_index_raises_data_error(
    tmp_path: Path,
) -> None:
    """Per spec 13.2: duplicate Image Index is fail-loud."""
    rows = FIXTURE_CSV.read_text().splitlines()
    # Duplicate the first data row (line index 1).
    bad = "\n".join([*rows, rows[1]]) + "\n"
    bad_path = tmp_path / "dup.csv"
    bad_path.write_text(bad)
    ingestor = NIHCsvIngestor()
    with pytest.raises(DataError, match=r"duplicate"):
        ingestor.ingest(bad_path)


def test_ingest_missing_required_column_raises_data_error(
    tmp_path: Path,
) -> None:
    """Per spec 2: header validation against canonical schema."""
    bad_path = tmp_path / "bad_header.csv"
    bad_path.write_text(
        "Image Index,Finding Labels,Patient ID\n"
        "00000001_000.png,Cardiomegaly,1\n"
    )
    ingestor = NIHCsvIngestor()
    with pytest.raises(DataError, match=r"header"):
        ingestor.ingest(bad_path)


def test_ingest_extra_columns_rejected(tmp_path: Path) -> None:
    """Per spec 2: header must equal the canonical schema exactly."""
    rows = FIXTURE_CSV.read_text().splitlines()
    header = rows[0] + ",ExtraCol"
    data = [r + ",x" for r in rows[1:]]
    bad_path = tmp_path / "extra_cols.csv"
    bad_path.write_text("\n".join([header, *data]) + "\n")
    ingestor = NIHCsvIngestor()
    with pytest.raises(DataError, match=r"header"):
        ingestor.ingest(bad_path)


def test_ingest_malformed_row_raises_data_error(tmp_path: Path) -> None:
    """A row with the wrong number of fields raises DataError."""
    rows = FIXTURE_CSV.read_text().splitlines()
    bad_row = "00000099_000.png,Cardiomegaly,0,99"  # missing fields
    bad_path = tmp_path / "malformed.csv"
    bad_path.write_text("\n".join([rows[0], bad_row]) + "\n")
    ingestor = NIHCsvIngestor()
    with pytest.raises(DataError):
        ingestor.ingest(bad_path)


def test_ingest_unknown_label_raises_data_error(tmp_path: Path) -> None:
    rows = FIXTURE_CSV.read_text().splitlines()
    bad_row = (
        "00000099_000.png,AlienFinding,0,99,30,M,PA,2500,2500,0.143,0.143"
    )
    bad_path = tmp_path / "alien.csv"
    bad_path.write_text("\n".join([rows[0], bad_row]) + "\n")
    ingestor = NIHCsvIngestor()
    with pytest.raises(DataError, match=r"AlienFinding"):
        ingestor.ingest(bad_path)


def test_ingest_empty_file_returns_empty_tuple(tmp_path: Path) -> None:
    """A header-only CSV is valid and returns an empty tuple."""
    rows = FIXTURE_CSV.read_text().splitlines()
    header_only = tmp_path / "header_only.csv"
    header_only.write_text(rows[0] + "\n")
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(header_only)
    assert records == ()


def test_ingest_uses_default_encoder_when_not_provided() -> None:
    """The constructor accepts a default-constructed encoder."""
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    assert len(records) == 16


def test_ingest_record_is_frozen() -> None:
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(FIXTURE_CSV)
    r = records[0]
    with pytest.raises(AttributeError):
        r.image_index = "mutated.png"  # type: ignore[misc]


def test_fixture_row_count_matches_csv() -> None:
    """Sanity: synthetic fixture really has 16 data rows + header."""
    with FIXTURE_CSV.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) == 17  # 1 header + 16 data


# ---------------------------------------------------------------------------
# Path-traversal / Image Index defence (review minor m3)
# ---------------------------------------------------------------------------


def test_ingest_rejects_image_index_with_forward_slash(tmp_path: Path) -> None:
    rows = FIXTURE_CSV.read_text().splitlines()
    bad_row = (
        "subdir/evil.png,Cardiomegaly,0,99,30,M,PA,2500,2500,0.143,0.143"
    )
    bad_path = tmp_path / "slash.csv"
    bad_path.write_text("\n".join([rows[0], bad_row]) + "\n")
    ingestor = NIHCsvIngestor()
    with pytest.raises(DataError, match=r"suspicious"):
        ingestor.ingest(bad_path)


def test_ingest_rejects_image_index_with_backslash(tmp_path: Path) -> None:
    rows = FIXTURE_CSV.read_text().splitlines()
    bad_row = (
        "subdir\\evil.png,Cardiomegaly,0,99,30,M,PA,2500,2500,0.143,0.143"
    )
    bad_path = tmp_path / "backslash.csv"
    bad_path.write_text("\n".join([rows[0], bad_row]) + "\n")
    ingestor = NIHCsvIngestor()
    with pytest.raises(DataError, match=r"suspicious"):
        ingestor.ingest(bad_path)


def test_ingest_rejects_image_index_literal_dotdot(tmp_path: Path) -> None:
    rows = FIXTURE_CSV.read_text().splitlines()
    bad_row = "..,Cardiomegaly,0,99,30,M,PA,2500,2500,0.143,0.143"
    bad_path = tmp_path / "dotdot.csv"
    bad_path.write_text("\n".join([rows[0], bad_row]) + "\n")
    ingestor = NIHCsvIngestor()
    with pytest.raises(DataError, match=r"suspicious"):
        ingestor.ingest(bad_path)


def test_ingest_accepts_image_index_with_dotdot_substring(tmp_path: Path) -> None:
    """``..foo.png`` is *not* a path traversal; only the literal ``..``
    component or explicit separators are blocked."""
    rows = FIXTURE_CSV.read_text().splitlines()
    benign_row = (
        "..foo.png,Cardiomegaly,0,99,30,M,PA,2500,2500,0.143,0.143"
    )
    csv_path = tmp_path / "dotdot_substr.csv"
    csv_path.write_text("\n".join([rows[0], benign_row]) + "\n")
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(csv_path)
    assert len(records) == 1
    assert records[0].image_index == "..foo.png"


# ---------------------------------------------------------------------------
# Blank-line behaviour (review major M7)
# ---------------------------------------------------------------------------


def test_ingest_blank_line_mid_file_raises_data_error(tmp_path: Path) -> None:
    """Per spec §13: blank lines mid-file are malformed CSV, fail loud."""
    rows = FIXTURE_CSV.read_text().splitlines()
    # Inject a blank line between header and first data row.
    blob = rows[0] + "\n\n" + "\n".join(rows[1:]) + "\n"
    bad_path = tmp_path / "blank_line.csv"
    bad_path.write_text(blob)
    ingestor = NIHCsvIngestor()
    with pytest.raises(DataError, match=r"blank line"):
        ingestor.ingest(bad_path)


def test_ingest_trailing_newlines_are_tolerated(tmp_path: Path) -> None:
    """Trailing newlines after the last row are an artifact of editors;
    csv.reader skips them, so ingestion must succeed."""
    blob = FIXTURE_CSV.read_text().rstrip("\n") + "\n\n\n"
    csv_path = tmp_path / "trailing_nl.csv"
    csv_path.write_text(blob)
    ingestor = NIHCsvIngestor()
    records = ingestor.ingest(csv_path)
    assert len(records) == 16
