"""Pure CSV + label-encoding adapter for the NIH ChestX-ray14 dataset.

This module is the Wave 2a slice of the NIH adapter (see
``harness/adapters/fs/docs/NIH_DATASET_SPEC.md`` sections 2, 3, 4.1, 7, 12,
and 13). It is intentionally I/O-restricted to text reads of the CSV
manifest:

* Zero PIL / numpy / pandas / sklearn / torch imports.
* Zero pixel decoding.
* Stdlib ``csv`` parsing only.
* Failures funnel through :class:`harness.domain.errors.DataError`.

Public surface:

* :data:`NIH14_LABELS` -- canonical 14-label vocabulary (Wang 2017 order).
* :data:`NIH14_INDEX` -- read-only label-to-index lookup.
* :data:`NIH_CSV_HEADER` -- positional 11-column header schema.
* :class:`NIHLabelEncoder` -- pipe-delimited string -> 14-tuple multi-hot.
* :class:`NIHCsvIngestor` -- CSV path -> ``tuple[NIHRecord, ...]``.
* :class:`NIHRecord` -- frozen-slots dataclass for one parsed row.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from harness.domain.errors import DataError

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Canonical 14-label NIH ChestX-ray vocabulary (Wang et al., 2017).
#: The index of label *i* in this tuple is its multi-hot bit position.
NIH14_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

#: Read-only ``label -> multi-hot index`` mapping derived from
#: :data:`NIH14_LABELS`. Wrapped in :class:`types.MappingProxyType` so callers
#: cannot mutate the canonical ordering at runtime.
NIH14_INDEX: Mapping[str, int] = MappingProxyType(
    {label: idx for idx, label in enumerate(NIH14_LABELS)}
)

#: Sentinel string in the ``Finding Labels`` column meaning "no positive
#: findings". Encodes to the all-zero multi-hot vector.
_NO_FINDING: str = "No Finding"

#: Canonical positional header tokens for ``Data_Entry_2017_v2020.csv``.
#: Each NIH bracketed header (``OriginalImage[Width,Height]`` and
#: ``OriginalImagePixelSpacing[x,y]``) is split into two tokens because the
#: comma inside the bracket is unescaped in the on-disk file.
NIH_CSV_HEADER: tuple[str, ...] = (
    "Image Index",
    "Finding Labels",
    "Follow-up #",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "View Position",
    "OriginalImage[Width",
    "Height]",
    "OriginalImagePixelSpacing[x",
    "y]",
)


# ---------------------------------------------------------------------------
# NIHRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NIHRecord:
    """One parsed NIH CSV row, sans image bytes.

    Field types match the spec's positional schema (§2). ``label_vector``
    is a 14-int multi-hot tuple in canonical NIH-14 order. Image dimensions
    and pixel spacings come from columns 7-10 of the CSV (the bracketed
    ``OriginalImage[Width,Height]`` and ``OriginalImagePixelSpacing[x,y]``
    headers split into 4 distinct positional fields).
    """

    image_index: str
    patient_id: str
    label_vector: tuple[int, ...]
    follow_up: int
    age: int
    gender: str
    view_position: str
    width: int
    height: int
    pixel_spacing_x: float
    pixel_spacing_y: float


# ---------------------------------------------------------------------------
# NIHLabelEncoder
# ---------------------------------------------------------------------------


class NIHLabelEncoder:
    """Pure pipe-string -> multi-hot encoder for the 14 NIH labels.

    Carries no state beyond the canonical label list. Whitespace around
    pipe-separated tokens is tolerated (defensive parsing per spec §12).
    Empty input, unknown labels, and the ``No Finding`` sentinel are
    handled per spec §13.1.
    """

    __slots__ = ("_index",)

    def __init__(self) -> None:
        self._index: Mapping[str, int] = NIH14_INDEX

    def encode(self, finding_labels_str: str | None) -> tuple[int, ...]:
        """Return the 14-int multi-hot vector for ``finding_labels_str``.

        * ``"No Finding"`` -> ``(0,) * 14``.
        * ``"Cardiomegaly|Effusion"`` -> tuple with bits 1 and 2 set.
        * Whitespace-padded tokens are stripped before lookup.
        * Duplicate tokens within a row are idempotent.
        * ``None`` or an empty/whitespace-only string raises
          :class:`DataError`. ``None`` is accepted in the type signature
          because callers may forward an absent CSV cell verbatim; the
          encoder normalises that to a single error path rather than
          requiring callers to branch.
        * An unknown label raises :class:`DataError` naming the label.
        """
        if finding_labels_str is None or finding_labels_str.strip() == "":
            raise DataError("empty Finding Labels string")

        stripped = finding_labels_str.strip()
        if stripped == _NO_FINDING:
            return (0,) * 14

        bits: list[int] = [0] * 14
        for raw_token in stripped.split("|"):
            token = raw_token.strip()
            if token == "":
                raise DataError(
                    f"empty token in Finding Labels: {finding_labels_str!r}"
                )
            if token == _NO_FINDING:
                # Per spec §3, "No Finding" is exclusive: it must never be
                # mixed with positive findings.
                raise DataError(
                    f"'No Finding' cannot be combined with other labels: "
                    f"{finding_labels_str!r}"
                )
            idx = self._index.get(token)
            if idx is None:
                raise DataError(
                    f"unknown NIH-14 label {token!r} in {finding_labels_str!r}"
                )
            bits[idx] = 1
        return tuple(bits)


# ---------------------------------------------------------------------------
# NIHCsvIngestor
# ---------------------------------------------------------------------------


class NIHCsvIngestor:
    """Parse ``Data_Entry_2017_v2020.csv`` into a tuple of :class:`NIHRecord`.

    Pure-text I/O only: opens the CSV file, validates the header against
    :data:`NIH_CSV_HEADER`, and emits one :class:`NIHRecord` per data row in
    CSV row order. Defends against duplicate ``Image Index`` values and
    malformed rows (spec §13.2). All failures raise
    :class:`DataError` so callers can funnel adapter errors through
    :class:`harness.domain.errors.HarnessError`.
    """

    __slots__ = ("_encoder",)

    def __init__(self, encoder: NIHLabelEncoder | None = None) -> None:
        self._encoder: NIHLabelEncoder = (
            encoder if encoder is not None else NIHLabelEncoder()
        )

    def ingest(self, csv_path: Path) -> tuple[NIHRecord, ...]:
        """Read ``csv_path`` and return the tuple of parsed NIH records."""
        try:
            handle = csv_path.open(newline="", encoding="utf-8")
        except OSError as exc:
            raise DataError(f"cannot open NIH CSV {csv_path!s}: {exc}") from exc

        with handle as f:
            reader = csv.reader(f)
            try:
                header_row = next(reader)
            except StopIteration:
                raise DataError(
                    f"NIH CSV {csv_path!s} is empty (no header row)"
                ) from None

            self._validate_header(header_row, csv_path)

            records: list[NIHRecord] = []
            seen: set[str] = set()
            expected_cols = len(NIH_CSV_HEADER)
            # Defer blank-row diagnosis: a blank row is only malformed if a
            # non-blank row follows it (mid-file gap). Trailing blank rows
            # from a final newline are silently dropped (spec §13).
            pending_blank_row_no: int | None = None

            for row_no, row in enumerate(reader, start=2):
                if not row:
                    if pending_blank_row_no is None:
                        pending_blank_row_no = row_no
                    continue
                if pending_blank_row_no is not None:
                    raise DataError(
                        f"NIH CSV {csv_path!s} row "
                        f"{pending_blank_row_no}: blank line"
                    )
                if len(row) != expected_cols:
                    raise DataError(
                        f"NIH CSV {csv_path!s} row {row_no}: expected "
                        f"{expected_cols} fields, got {len(row)}"
                    )
                record = self._parse_row(row, row_no, csv_path)
                if record.image_index in seen:
                    raise DataError(
                        f"duplicate Image Index {record.image_index!r} "
                        f"in {csv_path!s} (row {row_no})"
                    )
                seen.add(record.image_index)
                records.append(record)

        return tuple(records)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _validate_header(header_row: list[str], csv_path: Path) -> None:
        actual = tuple(header_row)
        if actual != NIH_CSV_HEADER:
            raise DataError(
                f"NIH CSV {csv_path!s} has invalid header: "
                f"expected {NIH_CSV_HEADER!r}, got {actual!r}"
            )

    def _parse_row(
        self, row: list[str], row_no: int, csv_path: Path
    ) -> NIHRecord:
        image_index = row[0]
        # Path-traversal defence: reject anything that could escape
        # ``images_dir`` or refer to a directory rather than a flat file.
        # ``..foo.png`` is *not* a traversal — only the literal ``..``
        # component and explicit path separators are.
        if (
            "/" in image_index
            or "\\" in image_index
            or image_index in {".", ".."}
        ):
            raise DataError(
                f"NIH CSV {csv_path!s} row {row_no}: suspicious "
                f"Image Index {image_index!r}"
            )

        finding_labels = row[1]
        try:
            label_vector = self._encoder.encode(finding_labels)
        except DataError as exc:
            raise DataError(
                f"NIH CSV {csv_path!s} row {row_no}: {exc}"
            ) from exc

        try:
            follow_up = int(row[2])
            patient_id_int = int(row[3])
            age = int(row[4])
            width = int(row[7])
            height = int(row[8])
        except ValueError as exc:
            raise DataError(
                f"NIH CSV {csv_path!s} row {row_no}: integer parse failed: "
                f"{exc}"
            ) from exc

        try:
            pixel_spacing_x = float(row[9])
            pixel_spacing_y = float(row[10])
        except ValueError as exc:
            raise DataError(
                f"NIH CSV {csv_path!s} row {row_no}: float parse failed: "
                f"{exc}"
            ) from exc

        gender = row[5]
        view_position = row[6]

        return NIHRecord(
            image_index=image_index,
            patient_id=str(patient_id_int),
            label_vector=label_vector,
            follow_up=follow_up,
            age=age,
            gender=gender,
            view_position=view_position,
            width=width,
            height=height,
            pixel_spacing_x=pixel_spacing_x,
            pixel_spacing_y=pixel_spacing_y,
        )
