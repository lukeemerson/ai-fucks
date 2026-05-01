# NIH ChestX-ray14 Dataset Adapter — Design Spec

**Status:** v1 spec, source of truth for Wave 2 + Wave 3 implementation agents.
**Audience:** every agent contributing to `harness/adapters/fs/nih_*` and
`tests/harness/{unit,contract,integration}/...nih...`.
**Scope:** filesystem-backed adapter that satisfies
`harness.ports.dataset.DatasetPort` against the public NIH ChestX-ray14 release
(Wang et al., 2017): `Data_Entry_2017_v2020.csv` plus PNG images on local disk.
**Architectural anchor:** ARCHITECTURE.md §2.1 (layering), §3.1 (`Sample`),
§4.1 (`DatasetPort`), §13 (v1 adopted surface). When this spec disagrees with
the implementation it is the spec that is wrong; open a follow-up spec PR
before code changes.

This document is the contract Wave 2 and Wave 3 agents follow. It deliberately
contains **no executable code** beyond minimal signature snippets reproduced
verbatim from the existing port.

---

## Table of Contents

1. [Port Contract Surface](#1-port-contract-surface)
2. [CSV Schema](#2-csv-schema)
3. [Canonical NIH-14 Label List](#3-canonical-nih-14-label-list)
4. [Module Layout](#4-module-layout)
5. [Image Processing Pipeline](#5-image-processing-pipeline)
6. [Patient-Level Metadata](#6-patient-level-metadata)
7. [`Sample` Field Mapping](#7-sample-field-mapping)
8. [Configuration Object](#8-configuration-object)
9. [Test Fixture Design](#9-test-fixture-design)
10. [Contract Test Integration](#10-contract-test-integration)
11. [Smoke Test Against Real On-Disk Subset](#11-smoke-test-against-real-on-disk-subset)
12. [TDD Plan (Wave Ordering)](#12-tdd-plan-wave-ordering)
13. [Edge Cases & Policies](#13-edge-cases--policies)
14. [Non-Goals for v1](#14-non-goals-for-v1)
15. [Spec Deviations from `DatasetPort`](#15-spec-deviations-from-datasetport)

---

## 1. Port Contract Surface

The adapter must satisfy the existing protocol exactly as defined in
`harness/ports/dataset.py`. Reproduced verbatim from that file (lines 18–33):

```python
@runtime_checkable
class DatasetPort(Protocol):
    """Source-of-truth port for dataset loading and image resolution."""

    def load(self) -> Dataset:
        """Return the full dataset (samples + label vocabulary)."""
        ...

    def get_image_bytes(self, image_ref: str) -> bytes:
        """Resolve an opaque ``image_ref`` to raw image bytes.

        Adapters must raise a :class:`harness.domain.errors.HarnessError`
        subclass (typically :class:`~harness.domain.errors.DataError`) if the
        reference is unknown or the underlying resource is unreadable.
        """
        ...
```

Key implications:

- `load()` returns `harness.domain.types.Dataset`. Per `harness/domain/types.py`
  lines 49–64 the `Dataset` invariant is that every `sample.labels` length
  equals `len(label_names)`. With the canonical NIH-14 vocabulary this means
  every multi-hot vector has length 14.
- `get_image_bytes(image_ref)` returns the **raw on-disk bytes** for the
  referenced image, not a decoded numpy array. The contract surface is
  intentionally narrow so backbones (sklearn, torch) own decoding and tensor
  conversion. The NIH adapter therefore decodes/normalizes images **for its
  own caches and any optional convenience accessors**, but `get_image_bytes`
  must return the unmodified PNG payload as written by NIH.
- Failures must funnel through `HarnessError`. Per `harness/domain/errors.py`
  the appropriate subclass for "unknown ref / unreadable file" is `DataError`;
  for adapter-construction or third-party failures it is `AdapterError`.
- The protocol is `@runtime_checkable`, so the adapter need only be
  structurally compatible; explicit inheritance is not required (and is not
  used by the in-memory fake at `harness/adapters/fakes/dataset.py`).

The contract test suite that the adapter must pass is
`tests/harness/contract/test_dataset_contract.py`. It asserts: `load` returns
a `Dataset`; two consecutive `load` calls produce the same `name`,
`label_names`, and ordered `sample_id` sequence; every `sample.labels` length
matches `len(ds.label_names)`; `get_image_bytes` returns `bytes` for a known
ref; consecutive calls for the same ref return byte-equal payloads; an unknown
ref raises a `HarnessError` subclass.

---

## 2. CSV Schema

The on-disk metadata file is `Data_Entry_2017_v2020.csv` at the repository
root (112,121 data rows + 1 header row). The header line, reproduced
verbatim:

```
Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]
```

Note the bracketed names: `OriginalImage[Width,Height]` is a **single column
header that itself contains a comma**. NIH's CSV writer escapes nothing, so
naive `csv.reader` splits would over-count fields. In practice the file
contains commas inside that header but the bracketed values become **two
separate columns** in the data rows (i.e. width and height are separated by a
comma at the data level even though the header presents them as one bracketed
group). The same is true for `OriginalImagePixelSpacing[x,y]`. The CSV
ingestor must therefore treat the header as a fixed positional schema rather
than name-matching against bracketed headers.

**Authoritative positional schema (zero-indexed):**

| Idx | Header (verbatim)                  | Type   | Notes                                                                |
| --- | ---------------------------------- | ------ | -------------------------------------------------------------------- |
|  0  | `Image Index`                      | str    | PNG filename, e.g. `00000001_000.png`. Unique per row in NIH v2020.  |
|  1  | `Finding Labels`                   | str    | Pipe-delimited multi-label, e.g. `Cardiomegaly\|Effusion`. `No Finding` is the all-negative sentinel. |
|  2  | `Follow-up #`                      | int    | Visit index for this patient; not used by `Sample`, retained in `metadata`. |
|  3  | `Patient ID`                       | int    | Integer in the CSV, kept as **string** in `Sample.patient_id`.       |
|  4  | `Patient Age`                      | int    | Decimal years.                                                       |
|  5  | `Patient Gender`                   | str    | `M` / `F`.                                                           |
|  6  | `View Position`                    | str    | `PA` / `AP`.                                                         |
|  7  | `OriginalImage[Width`              | int    | Width in pixels (header is split).                                   |
|  8  | `Height]`                          | int    | Height in pixels (header is split).                                  |
|  9  | `OriginalImagePixelSpacing[x`      | float  | x spacing.                                                           |
| 10  | `y]`                               | float  | y spacing.                                                           |

The first three data rows (verbatim) confirm the layout:

```
00000001_000.png,Cardiomegaly,0,1,57,M,PA,2682,2749,0.14300000000000002,0.14300000000000002
00000001_001.png,Cardiomegaly|Emphysema,1,1,58,M,PA,2894,2729,0.14300000000000002,0.14300000000000002
00000001_002.png,Cardiomegaly|Effusion,2,1,58,M,PA,2500,2048,0.168,0.168
```

**Parsing decisions:**

- Use `csv.reader` from the stdlib with default dialect. Do **not** depend on
  pandas in `nih_csv.py` (keeps the harness numpy-only at this layer).
- After reading the header line, the ingestor must validate that the header
  exactly equals the canonical 11-token positional schema above. A mismatch
  raises `DataError` with the diff, naming `nih_csv.NIHCsvIngestor`.
- `Patient ID` is read as `int`, then **stored as `str(int_value)`** in
  `Sample.patient_id` so downstream splitters (per ARCHITECTURE.md §3.1, all
  IDs are `str`) can use lexicographic equality. Storing the raw token without
  parsing is forbidden — the int-round-trip normalises any whitespace.
- `Image Index` is preserved verbatim as `Sample.sample_id`.
- `Patient Age`, `Patient Gender`, `View Position`, `Follow-up #`, image
  dimensions, and pixel spacings are stored in `Sample.metadata` as a frozen
  `Mapping[str, str]`. All values are coerced to `str` at the boundary. The
  metadata key set is documented in §7.

---

## 3. Canonical NIH-14 Label List

Per Wang et al. (2017) the 14 NIH-14 disease labels in canonical order are:

1. `Atelectasis`
2. `Cardiomegaly`
3. `Effusion`
4. `Infiltration`
5. `Mass`
6. `Nodule`
7. `Pneumonia`
8. `Pneumothorax`
9. `Consolidation`
10. `Edema`
11. `Emphysema`
12. `Fibrosis`
13. `Pleural_Thickening`
14. `Hernia`

This ordering is **immutable** for v1 and is the order in which `label_names`
appears in the returned `Dataset`. The multi-hot vector index for label *i*
matches its 1-based ordinal in the list above (zero-indexed in code: index 0
is `Atelectasis`, index 13 is `Hernia`).

Determinism notes:

- The list is exposed as a module-level constant `NIH14_LABELS:
  tuple[str, ...]` in `harness/adapters/fs/nih_csv.py`. It is the single
  source of truth; tests import from this constant rather than re-typing.
- A second constant `NIH14_INDEX: Mapping[str, int]` maps canonical label
  string → multi-hot index. It is constructed once at import and exposed as
  a `MappingProxyType` so callers cannot mutate it.
- `No Finding` is **not** one of the 14. It is an exclusive sentinel meaning
  "the multi-hot vector is all zeros." It must never appear as an entry in
  `label_names`; it must never set any bit. A row whose `Finding Labels` is
  exactly `No Finding` produces the multi-hot vector `(0,) * 14`.
- The label-name strings are exact-case and use `Pleural_Thickening` with
  the underscore (per Wang 2017 and the on-disk CSV). Adapters must not
  normalise to lowercase or replace the underscore.

---

## 4. Module Layout

Three files under `harness/adapters/fs/`:

```
harness/adapters/fs/
  nih_dataset.py    # NIHDataset + NIHDatasetConfig — composition only
  nih_csv.py        # NIHCsvIngestor + NIHLabelEncoder — pure CSV → records
  nih_images.py     # NIHImageLoader + ImageCache — pure bytes/array I/O
```

### 4.1 `nih_csv.py` (pure)

Responsibilities:

- `NIHCsvIngestor` — reads the CSV, validates the header, yields a `tuple` of
  immutable `NIHCsvRecord` dataclasses (one per row). Pure function of (path,
  optional `subset_filter`). Zero PIL/pandas/numpy imports — stdlib `csv` only.
- `NIHCsvRecord` — internal frozen dataclass with the 11 typed fields plus
  the parsed `tuple[int, ...]` multi-hot vector. Not part of the public
  surface; consumed only by `nih_dataset.py`.
- `NIHLabelEncoder` — pure callable wrapper translating
  `pipe_string -> tuple[int, ...]` of length 14. Carries no state beyond the
  canonical label list. Methods (signatures only):
  ```python
  class NIHLabelEncoder:
      def encode(self, finding_labels: str) -> tuple[int, ...]: ...
      def decode(self, multi_hot: tuple[int, ...]) -> tuple[str, ...]: ...
  ```
- Module-level constants: `NIH14_LABELS`, `NIH14_INDEX`,
  `NIH_CSV_HEADER: tuple[str, ...]` (the exact 11-token positional schema
  used for validation).

Forbidden imports here: `PIL`, `Pillow`, `pandas`, `cv2`, `numpy` (records
are plain Python tuples; numpy enters at the image layer only). Allowed:
`csv`, `dataclasses`, `pathlib`, `typing`, `types`, `collections.abc`, and
`harness.domain.errors`.

### 4.2 `nih_images.py` (pure)

Responsibilities:

- `NIHImageLoader` — given an absolute PNG path, reads the file once and
  returns: (a) the raw bytes (for `DatasetPort.get_image_bytes`), and
  (b) a decoded `NDArray[np.float32]` array (for cache pre-warming and any
  internal smoke checks). Decoding uses Pillow exclusively. Adapters
  downstream (the torch backbone) re-decode from bytes; this loader's
  decoded output is **not** part of the port surface.
- `ImageCache` — bounded LRU cache, keyed by absolute path string, holding
  decoded `NDArray[np.float32]` arrays. Eviction policy: classic LRU with
  configurable `cache_size` (default 1024). When `disk_cache_dir` is
  configured, on cache miss the loader checks for `<sha256(abs_path)>.npy`
  in that directory and uses it; on cache load from PNG it writes the same
  `.npy` for next-run reuse. Disk cache writes are atomic (`tmp + rename`).

Forbidden imports here: `csv`, `pandas`, `harness.adapters.fs.nih_csv`. The
image module knows nothing about labels, patient IDs, or the CSV schema.
Allowed: `pathlib`, `typing`, `hashlib`, `numpy`, `PIL`/`Pillow`, and
`harness.domain.errors`.

### 4.3 `nih_dataset.py` (composition)

Responsibilities:

- `NIHDataset` — top-level adapter implementing `DatasetPort`. Composes
  `NIHCsvIngestor` + `NIHLabelEncoder` + `NIHImageLoader` + `ImageCache`.
  Owns the lifecycle: in `__init__` it validates inputs against config,
  walks the CSV, optionally checks each referenced PNG exists (per
  `strict_missing_images`), and constructs the immutable `Dataset`.
- `NIHDatasetConfig` — configuration dataclass (see §8).

This module is the only one that imports both `nih_csv` and `nih_images`.
Per ARCHITECTURE.md §2.1 ("adapters may not import from another adapter
package") it must not import from `harness.adapters.sklearn`,
`harness.adapters.torch`, or `harness.adapters.fakes`.

### 4.4 Image-ref convention

`Sample.image_ref` (the opaque token consumed by `get_image_bytes`) is the
**absolute filesystem path** of the PNG, expressed as a string. Rationale:

- `get_image_bytes` must be O(1) and deterministic. Re-resolving relative
  paths against `images_dir` on every call is brittle if the working
  directory changes between tests; storing absolute paths once at `load`
  time avoids that class of bug.
- The fake's convention (`ref://N`) is opaque-by-construction; the
  filesystem adapter's convention is opaque-but-resolvable. Both satisfy
  the port — the port treats `image_ref` as opaque from the *consumer's*
  perspective.
- The adapter retains an internal `frozenset[str]` of known refs to
  short-circuit the unknown-ref case with `DataError` before touching
  the disk.

---

## 5. Image Processing Pipeline

Two distinct concerns sit in this layer; they must not be conflated.

### 5.1 The port surface (`get_image_bytes`)

Returns the **raw PNG bytes from disk**. No decoding, no normalisation, no
resizing. Reads via `Path.read_bytes()`. Wraps `OSError` /
`FileNotFoundError` in `DataError` with the absolute path. This is the only
output a backbone-side decoder ever sees in production.

### 5.2 The internal decoded representation (`NIHImageLoader.load_array`)

Used by the cache and by smoke tests. **Not** exposed through `DatasetPort`.
Pipeline:

1. `Image.open(path)` via Pillow.
2. Convert to single-channel grayscale via `image.convert("L")`. NIH images
   are grayscale on the wire but some are saved with a 3-channel palette;
   forcing `L` normalises the variation.
3. Resize to `config.image_size` (default `(224, 224)`) using
   `Image.Resampling.BILINEAR`. Default size matches ResNet50 / DenseNet121
   inputs which are the v1 publication-run candidates.
4. Convert to `numpy.ndarray` via `numpy.asarray(...)`. The `uint8` array is
   then cast to `numpy.float32` and divided by `255.0`.
5. Returned shape: `(H, W, 1)` (channel-last, single-channel). The trailing
   axis is mandatory so downstream backbones can broadcast or replicate to
   3 channels without ambiguity. Returning `(H, W)` is forbidden.
6. Returned dtype: `numpy.float32`. Returned value range: `[0.0, 1.0]`
   inclusive. Any pixel outside this range is a defect.
7. **No ImageNet mean/std subtraction.** That is the backbone adapter's
   responsibility (and depends on which backbone — ImageNet vs RadImageNet
   have different mean/std). The contract of `NIHImageLoader` is exactly:
   bytes-on-disk → clean float32 grayscale array in `[0, 1]` at the
   configured size.

### 5.3 Caching

- In-memory: `ImageCache` is a bounded LRU with `cache_size` entries
  (default 1024). Cache key is the absolute path string. Cache value is the
  `(H, W, 1)` `float32` array. Eviction is least-recently-used with
  `OrderedDict.move_to_end` semantics.
- On-disk (optional): when `config.disk_cache_dir` is set, missed cache
  reads first look for `<disk_cache_dir>/<sha256(abs_path)>.npy`. Cache
  writes go through `tmp + os.replace` for atomicity. The on-disk cache is
  invalidated implicitly when `image_size` changes — the spec recommends
  using a per-size sub-directory (`<disk_cache_dir>/h<H>_w<W>/`) so two
  configurations cannot share entries.
- Hits are tracked but not exposed publicly in v1; an adapter-private
  counter (`_hits`, `_misses`) is sufficient for testing.
- Cache invalidation on file mtime change is **not** required for v1. NIH
  images are immutable on disk.

---

## 6. Patient-Level Metadata

Per `harness/domain/types.py` line 43 `Sample.patient_id` is `str`. Per the
NIH CSV the source value is an integer.

Mapping rule:

- Read CSV column `Patient ID` as `int`.
- Store `str(int_value)` (no zero-padding) in `Sample.patient_id`.

This is intentionally compatible with the existing
`IterativeStratifiedPatientSplitter` (sklearn adapter), which performs
patient-grouped iterative stratification on the string `patient_id` field.
ARCHITECTURE.md §3.3 splitter invariant requires that "no patient appears in
more than one set"; with NIH the patient grouping has on the order of 30,800
unique patients across 112,121 rows, so the splitter has ample signal.

The patient-level grouping is **not** performed by the dataset adapter. The
adapter's job ends at producing flat samples with correct `patient_id`
strings; the splitter does the grouping.

---

## 7. `Sample` Field Mapping

The full mapping for each NIH CSV row → one `Sample` (per
`harness/domain/types.py` lines 38–46):

| `Sample` field | Source                                     | Notes                                              |
| -------------- | ------------------------------------------ | -------------------------------------------------- |
| `sample_id`    | CSV `Image Index` (verbatim)               | E.g. `00000001_000.png`. Unique within the loaded subset. |
| `patient_id`   | CSV `Patient ID` parsed as int → `str(...)`| `"1"`, `"2"`, ... — no padding.                    |
| `image_ref`    | `str(<images_dir>/<Image Index>.resolve())`| Absolute path string; opaque to downstream callers.|
| `labels`       | `NIHLabelEncoder.encode(Finding Labels)`   | `tuple[int, ...]` of length 14. `No Finding` → all zeros. |
| `metadata`     | Frozen `Mapping[str, str]` (see below)     | All values stringified.                            |

Required keys in `Sample.metadata` (alphabetical, all values `str`):

- `follow_up` — CSV `Follow-up #` as decimal string.
- `patient_age` — CSV `Patient Age` as decimal string.
- `patient_gender` — CSV `Patient Gender` (`M` or `F`).
- `pixel_spacing_x` — CSV `OriginalImagePixelSpacing[x` as decimal string.
- `pixel_spacing_y` — CSV `y]` as decimal string.
- `view_position` — CSV `View Position` (`PA` or `AP`).
- `width` — CSV `OriginalImage[Width` as decimal string.
- `height` — CSV `Height]` as decimal string.

The metadata mapping is wrapped in `types.MappingProxyType` so it satisfies
the read-only `Mapping[str, str]` contract and is hashable-friendly inside
the frozen `Sample` dataclass.

### 7.1 The image-loading question (a / b / c)

The spec considers three patterns for surfacing image data:

- **(a) Eager numpy array on every `load()` call** — `Sample.image_ref` is a
  path; callers fetch decoded arrays via a separate convenience accessor.
- **(b) Lazy callable** — `Sample.image_ref` is a path; calling
  `dataset.get_image_bytes(ref)` returns bytes, and decoding is the
  caller's job.
- **(c) Bytes that the backbone decodes** — same as (b) but the spec is
  explicit that backbones own decoding.

**v1 chooses (c).** Justification:

- The port `get_image_bytes` already returns `bytes`. Layering forbids the
  domain `Sample` from carrying numpy arrays (per §3.1 of ARCHITECTURE.md
  the sample is a frozen dataclass; numpy arrays are mutable and not
  hashable in any useful sense).
- (a) would inflate `Dataset` to ~5 GB for the full 112k corpus at 224×224
  float32, defeating the lazy-loading premise of `DatasetPort`.
- (b) is structurally identical to (c) for the dataset adapter; the
  difference is whether the backbone or the caller decodes. The codebase
  already places that responsibility in the backbone (see
  ARCHITECTURE.md §13.1 row "BackbonePort.extract").

**Tradeoff documented:** (c) means the NIH adapter decodes a PNG twice in
the worst case — once internally for cache pre-warming if the user opts in,
and once again by the backbone from bytes. The duplicate decode is bounded
to N-once-per-run because the backbone caches its own embeddings; the
dataset adapter's optional disk cache stores the *decoded* float32 array,
not the bytes, so it is the backbone's first-decode cost we are paying. The
cleanest contract trumps the duplicate-decode optimisation in v1.

---

## 8. Configuration Object

```python
@dataclass(frozen=True, slots=True)
class NIHDatasetConfig:
    csv_path: Path
    images_dir: Path
    image_size: tuple[int, int] = (224, 224)
    cache_size: int = 1024
    disk_cache_dir: Path | None = None
    subset_filter: Callable[[Mapping[str, str]], bool] | None = None
    strict_missing_images: bool = True
    name: str = "nih-cxr14"
```

Field semantics:

- `csv_path` — absolute path to `Data_Entry_2017_v2020.csv`. Resolved at
  config-construction; relative paths raise `ConfigError`.
- `images_dir` — directory containing all NIH PNGs as flat files (NIH
  ships them as `images_001/...` through `images_012/...`; the convention
  here is that the user has flattened them or symlinked into a single
  directory). The adapter does **not** walk subdirectories.
- `image_size` — `(H, W)` tuple. Default `(224, 224)` matches both
  ResNet50 and DenseNet121 inputs after grayscale-to-3-channel replication.
  Both dimensions must be positive; mismatch raises `ConfigError`.
- `cache_size` — bounded LRU size for in-memory decoded arrays. `0`
  disables the cache. Negative raises `ConfigError`.
- `disk_cache_dir` — optional absolute path. When set, the loader writes
  per-image `.npy` files for cross-run reuse. The adapter creates the
  directory (and the per-size subdirectory) at config validation if it
  does not exist; failure to create raises `AdapterError`.
- `subset_filter` — callable accepting a row's metadata mapping (the same
  mapping that ends up in `Sample.metadata`, but augmented with the raw
  `image_index` and `patient_id` keys before filtering) and returning
  `True` to keep, `False` to drop. Used for testing on partial data and
  for the smoke test (which filters to "rows whose images exist locally").
  Default `None` keeps every row.
- `strict_missing_images` — when `True` (default), the adapter raises
  `DataError` if any CSV row references a missing PNG. When `False`, the
  adapter logs the count of dropped rows via the standard `logging` module
  and silently filters them out of the returned `Dataset`. The smoke test
  in §11 sets this to `False`.
- `name` — passed through as `Dataset.name`. Default `"nih-cxr14"`. The
  `ExperimentConfig.dataset_name` must match.

`__post_init__` performs:

1. `csv_path.is_file()` else `ConfigError("csv not found")`.
2. `images_dir.is_dir()` else `ConfigError("images_dir not found")`.
3. `image_size[0] > 0 and image_size[1] > 0` else `ConfigError`.
4. `cache_size >= 0` else `ConfigError`.
5. If `disk_cache_dir` is set, `mkdir(parents=True, exist_ok=True)`; on
   failure raise `AdapterError` with the OS error chained.

---

## 9. Test Fixture Design

The Wave-1b agent builds a tiny synthetic fixture under
`tests/harness/fixtures/nih/` so contract and unit tests run deterministically
in well under a second. Layout:

```
tests/harness/fixtures/nih/
  Data_Entry_tiny.csv
  images/
    00000001_000.png
    00000001_001.png
    00000002_000.png
    00000003_000.png
    00000004_000.png
    00000005_000.png
    00000005_001.png
    00000006_000.png
    00000007_000.png
    00000008_000.png
  README.md
```

### 9.1 Fixture CSV

`Data_Entry_tiny.csv` has the exact same header line as the real CSV (verbatim
the line in §2 above) and 10 data rows. The 10 rows must collectively cover:

- At least one `No Finding` row (the all-zero sentinel case).
- At least one single-label row.
- At least one multi-label row with the pipe delimiter (e.g.
  `Cardiomegaly|Effusion`).
- At least one `Hernia` row (the rarest NIH-14 label, exercised so the
  encoder's last index is verified).
- At least one row per `View Position` value (`PA` and `AP`).
- Both `M` and `F` patients.
- 5 distinct `Patient ID` values, with two patients having multiple studies
  (used by the splitter contract test).
- All 14 NIH-14 labels collectively appear at least once across the 10 rows
  (the union of the labels is the full vocabulary). This is the gating
  invariant for the encoder smoke check.

### 9.2 Fixture PNGs

Each fixture PNG is a deterministic 16×16 grayscale image generated
programmatically (do **not** check in stock photos; the fixture must be
reproducible from a seed). The Wave-1b implementer:

- Uses a fixed seed (e.g. `0x4E494831` = "NIH1") to produce a deterministic
  pixel pattern per filename — for example, the lower 8 bits of
  `sha256(filename)[i]` for each pixel `i`. The result is byte-stable across
  platforms.
- Writes via `PIL.Image.fromarray(arr, mode="L").save(path)` to keep the file
  a pure 8-bit grayscale PNG.
- The fixture README.md documents the seeding scheme so the fixture can be
  regenerated.

### 9.3 Filename contract

The 10 filenames listed in §9 above are required exactly. The CSV must
reference exactly those 10 image indices, in that order. No extra PNGs in
`images/`; no orphan rows in the CSV. (Tests for the
`image-present-but-not-in-CSV` and `CSV-row-but-image-missing` cases use
`tmp_path` fixtures that copy a subset and delete a file, rather than
polluting the canonical fixture.)

---

## 10. Contract Test Integration

The contract suite at `tests/harness/contract/test_dataset_contract.py`
already defines `DatasetPortContract` (an abstract test class) and one
concrete subclass `TestInMemoryFakeDatasetContract`. Wave-3 appends a second
concrete subclass:

```python
class TestNIHDatasetContract(DatasetPortContract):
    @pytest.fixture
    def adapter(self) -> DatasetPort:
        fixture_root = Path(__file__).parent.parent / "fixtures" / "nih"
        config = NIHDatasetConfig(
            csv_path=fixture_root / "Data_Entry_tiny.csv",
            images_dir=fixture_root / "images",
            image_size=(8, 8),         # tiny for speed
            cache_size=4,              # exercise eviction
            disk_cache_dir=None,
            subset_filter=None,
            strict_missing_images=True,
        )
        return NIHDataset(config)
```

All six existing contract tests then run automatically against the NIH
adapter:

1. `test_load_returns_dataset` — passes by construction.
2. `test_load_is_deterministic` — passes because the CSV is sorted
   read-order-stable and the adapter caches its `Dataset` after first load.
3. `test_label_vector_lengths_match_label_names` — passes because every
   sample uses the canonical 14-label encoder.
4. `test_get_image_bytes_returns_bytes_for_known_ref` — passes; the bytes
   are the actual PNG payload.
5. `test_get_image_bytes_is_deterministic` — passes; two `Path.read_bytes`
   calls on the same file return byte-equal results.
6. `test_get_image_bytes_unknown_ref_raises_harness_error` — passes; the
   adapter checks `image_ref in self._known_refs` first and raises
   `DataError` (a `HarnessError` subclass) before touching the disk.

Performance budget for the NIH contract tests: **< 1 second total**. The
fixture is tiny (10 rows, 16×16 PNGs); first `load()` is bounded by 10 file
existence checks plus 10 CSV row parses.

---

## 11. Smoke Test Against Real On-Disk Subset

The user has approximately 4,999 of the 112,121 NIH PNGs locally under
`db-test_images/images/`. A smoke test exercises the adapter against this
real subset to catch real-world parsing bugs the synthetic fixture cannot.

Location: `tests/harness/integration/test_nih_dataset_smoke.py`.

Behaviour:

1. Skip if `Data_Entry_2017_v2020.csv` or `db-test_images/images/` is
   missing (so CI without the data does not fail).
2. Build `NIHDatasetConfig` with:
   - `csv_path` pointed at the real CSV.
   - `images_dir` pointed at `db-test_images/images/`.
   - `strict_missing_images=False` so missing PNGs are dropped silently
     (the local subset is ~4.5% of the full release).
   - `image_size=(64, 64)` (small enough to keep RAM cheap; the smoke test
     does not need 224×224 fidelity).
   - `cache_size=128`.
3. Call `dataset.load()`. Assert:
   - `len(ds.samples) >= 4_500` and `<= 5_500` (sanity range; the local
     directory contains ~4,999 PNGs but some may not be in the CSV).
   - `len(ds.label_names) == 14`.
   - The label-name tuple equals the canonical NIH-14 list.
4. Take the first 100 samples; for each call `dataset.get_image_bytes`.
   Assert each result is non-empty and starts with the PNG signature
   `b"\x89PNG"` (the first 4 bytes of the 8-byte PNG signature).
5. Aggregate the multi-hot vectors of the first 100 samples and assert:
   - At least 5 distinct labels appear (sanity — not all `No Finding`).
   - The per-label counts sum to a number > 0 (someone has *some* finding).
6. Mark `@pytest.mark.smoke`. The default CI invocation
   (`pytest tests/harness/unit tests/harness/contract`, per ARCHITECTURE.md
   §1.3) does not include this marker, keeping the unit + contract suite
   under 5 seconds.

The smoke test is non-deterministic in count (because the local image
subset can change between developer machines) but is deterministic in
**structure** (label vocabulary, byte signatures, label-presence sanity).

---

## 12. TDD Plan (Wave Ordering)

Wave 2 lands two parallel sub-PRs; Wave 3 composes them. Every step opens
with a failing test (per ARCHITECTURE.md §8.1).

### Wave 2a — `nih_csv.py` (CSV + label encoder)

Tests at `tests/harness/unit/adapters/fs/test_nih_csv.py` and
`test_nih_label_encoder.py`. Required cases:

**`NIHLabelEncoder`:**
- `encode("No Finding")` returns `(0,) * 14`.
- `encode("Cardiomegaly")` returns the one-hot vector with bit 1 set
  (canonical index 1).
- `encode("Atelectasis")` returns one-hot at index 0.
- `encode("Hernia")` returns one-hot at index 13.
- `encode("Cardiomegaly|Effusion")` returns the two-hot vector with bits
  1 and 2 set, regardless of pipe-token order
  (`Effusion|Cardiomegaly` produces the same vector).
- `encode("")` raises `DataError("empty Finding Labels")`.
- Whitespace tolerance: `encode(" Cardiomegaly | Effusion ")` strips and
  encodes correctly (NIH's CSV is tight but defensive parsing prevents
  one class of foot-gun).
- `decode(encode(s)) == sorted_canonical_labels(s)` for every test string
  above (round-trip property).
- **Unknown-label policy:** `encode("AlienFinding")` raises `DataError`
  with the unknown label string included in the message. (Policy choice
  documented in §13.1.)

**`NIHCsvIngestor`:**
- Header validation: a CSV whose first line differs from
  `NIH_CSV_HEADER` raises `DataError` with the diff.
- 10-row fixture CSV → 10 records with correct types
  (`Image Index` str, `Patient ID` str, `labels` length 14).
- `subset_filter` is applied in row order; rows for which the filter
  returns `False` do not appear in output.
- An `Image Index` containing path separators (`../etc/passwd`) raises
  `DataError` with a clear message — defence against malformed CSV.
- Empty file (header only, no rows) returns an empty tuple of records,
  no exception.

### Wave 2b — `nih_images.py` (loader + cache)

Tests at `tests/harness/unit/adapters/fs/test_nih_image_loader.py` and
`test_image_cache.py`. Required cases:

**`NIHImageLoader.load_array`:**
- Input: a fixture 16×16 grayscale PNG. Output shape `(H, W, 1)` matches
  the configured `image_size`.
- Output dtype is `numpy.float32`.
- Output value range is `[0.0, 1.0]` inclusive (assert via `min` and
  `max`).
- Resize correctness: a fixture PNG with a single white pixel at (3, 5)
  resized to `(8, 8)` produces a peak in the upper-left quadrant
  consistent with bilinear interpolation. (Soft assertion: argmax row/col
  in the upper half.)
- Idempotency: calling `load_array(p)` twice returns equal arrays.
- A 3-channel PNG is correctly converted to single-channel grayscale
  (output last-axis size is 1).
- Missing file: `load_array("/no/such/path.png")` raises `DataError`.
- Corrupt PNG: a file whose contents are 8 random bytes raises
  `DataError` with the underlying Pillow message chained.

**`NIHImageLoader.load_bytes`:**
- Returns the unmodified bytes from disk for an existing PNG.
- Two consecutive calls on the same path return byte-equal payloads.
- Missing file raises `DataError`.

**`ImageCache`:**
- Cache hit: `get` after `put` returns the exact same array object (or
  an equal copy — implementation choice, but documented).
- Cache miss returns `None` (or raises `KeyError` — pick one and assert).
- LRU eviction: with `cache_size=2`, putting three distinct keys in
  succession evicts the first; the second `get` for the first key is a
  miss.
- `cache_size=0` disables caching; every `get` is a miss; `put` is a
  no-op.
- Disk-cache round-trip: with a `disk_cache_dir`, `put` writes a `.npy`
  file; clearing the in-memory cache and calling `get` reads the array
  back from disk. The round-tripped array equals the original.

### Wave 3 — `nih_dataset.py` (composition)

Tests:
1. Unit tests at `tests/harness/unit/adapters/fs/test_nih_dataset.py`:
   - Sample count equals fixture row count.
   - Label vocabulary equals canonical NIH-14 tuple.
   - First sample has correct `sample_id`, `patient_id`, and `image_ref`
     equal to the absolute fixture path.
   - `metadata` keys equal the documented set in §7.
   - `strict_missing_images=False` drops a row whose PNG is absent and
     produces a `Dataset` with the surviving samples.
   - `strict_missing_images=True` with a missing PNG raises `DataError`.
   - `subset_filter` is honoured (e.g. drop all rows with `view_position`
     != `"PA"`; assert all surviving samples have `metadata["view_position"]
     == "PA"`).
   - Deterministic ordering: `tuple(s.sample_id for s in ds.samples)`
     equals the CSV row order.
2. Contract test at
   `tests/harness/contract/test_dataset_contract.py::TestNIHDatasetContract`
   passes (per §10).
3. Smoke test at
   `tests/harness/integration/test_nih_dataset_smoke.py` passes when the
   real subset is on disk (per §11).

---

## 13. Edge Cases & Policies

### 13.1 Unknown labels

If a future NIH CSV release adds a 15th label, encountering it during
encoding **raises `DataError`** with the unknown label string included.
Rationale: silent ignore would corrupt every multi-hot vector for affected
rows without any visible signal; warn-and-skip would still produce a
mis-shaped vector for the row. Failing loudly forces the user to either
update the canonical list (a deliberate spec change) or filter the row via
`subset_filter`.

### 13.2 Duplicate `Image Index`

The NIH v2020 CSV does not contain duplicate `Image Index` values. The
adapter still defends against it: a duplicate causes
`DataError("duplicate Image Index: <name>")` from `NIHDataset.__init__`.
This is a fail-loud policy because downstream `Sample.sample_id` must be
unique within a `Dataset` (the splitter assumes it).

### 13.3 Image present but not in CSV

The CSV is the authoritative index. Files in `images_dir` that are not
referenced by any CSV row are **silently ignored**. The adapter does not
walk the directory at all; it only opens files referenced by the CSV.

### 13.4 CSV row but image missing

Behaviour is controlled by `config.strict_missing_images`:

- `True` (default): `NIHDataset.__init__` raises `DataError` listing the
  first up-to-20 missing filenames and the total missing count.
- `False`: the row is dropped from the loaded `Dataset`. The adapter logs
  `dropped {N} CSV rows: missing image files` at INFO level via the
  standard `logging` module. The dropped count is not part of the public
  surface; callers must re-count if they care.

Policy rationale: production runs use `True` (catch data corruption
loudly); developer smoke runs against the local 5k subset use `False`.

### 13.5 Patient with mixed labels across studies

Each CSV row produces its own `Sample`. A patient with three studies
(say, follow-up #0 with `No Finding`, follow-up #1 with `Cardiomegaly`,
follow-up #2 with `Cardiomegaly|Effusion`) yields three separate
`Sample` objects sharing the same `patient_id`. Patient grouping happens
at split time inside the splitter, not in the dataset adapter.

### 13.6 Images with palette mode or 16-bit depth

NIH PNGs are 8-bit grayscale on the wire, but a small number ship as
palette-mode PNGs. Pillow's `convert("L")` call handles both. The
loader does **not** re-quantise from higher bit depth; if Pillow returns
a 16-bit array the float32 normalisation `arr / 255.0` would push values
above 1.0. Defensive policy: after normalisation, `np.clip(arr, 0.0, 1.0,
out=arr)`. This is an explicit defence-in-depth; in practice `convert("L")`
already maps to 8-bit.

### 13.7 Working-directory drift

`Sample.image_ref` is always an absolute path string. Tests run from
random working directories (`pytest`'s `tmp_path` and `chdir`) must
still resolve image bytes correctly. This is enforced by the
`NIHDatasetConfig.__post_init__` validator that resolves `images_dir`
to an absolute path before storage.

### 13.8 Concurrency

The adapter is single-threaded. `ImageCache` is **not** thread-safe; if
a future caller wants concurrent access they must build a thread-safe
wrapper or use the disk cache only. Multi-process loading is explicitly
deferred (see §14).

---

## 14. Non-Goals for v1

Explicitly out of scope for the NIH adapter v1; downstream agents must
not smuggle these in:

- **DICOM support.** NIH ChestX-ray14 ships PNG; DICOM input belongs in
  v1.1 alongside MIMIC-CXR / CheXpert support, gated behind a separate
  adapter (`harness/adapters/fs/dicom_dataset.py`).
- **Image augmentation in the loader.** Random crops, flips, rotations,
  contrast jitter — none of these belong here. Augmentation is a
  train-time concern that lives in the head/backbone adapter or in a
  future `AugmentationPort`. The dataset adapter is deterministic by
  contract.
- **Multi-process / multi-threaded loading.** ResNet50 inference on
  4,999 images at 224×224 finishes in tens of seconds on Apple MPS;
  full 112k finishes in ~1.5 hours. The single-threaded loader is
  comfortably below that wall-clock floor. A future
  `harness/adapters/torch/dataset.py` may wrap this adapter in a
  `torch.utils.data.DataLoader` with workers.
- **Streaming / lazy `Dataset`.** v1 returns the entire `Dataset` from
  `load()`. With 112k samples × ~200 bytes per `Sample` that is ~22 MB
  of in-memory metadata, well within budget.
- **Network I/O.** No downloading, no kaggle CLI, no S3. The user
  supplies `csv_path` and `images_dir` as already-on-disk paths.
- **Label-policy hooks.** The 14-label canonical list is fixed.
  Renaming `Pleural_Thickening` to `Pleural Thickening`, collapsing
  `Mass` and `Nodule`, or adding a 15th label all require a spec PR.
- **Bounding-box ingestion.** NIH's `BBox_List_2017.csv` is a separate
  file and a separate adapter (`nih_bboxes.py`, deferred to v1.1 if
  detection becomes a goal).

---

## 15. Spec Deviations from `DatasetPort`

**None.** The NIH adapter implements the `DatasetPort` protocol exactly
as defined at `harness/ports/dataset.py` lines 18–33. Specifically:

- `load() -> Dataset` returns a fully-populated `Dataset` whose
  `label_names` is the canonical 14-label tuple and whose `samples` are
  the CSV-derived `Sample` objects in CSV row order.
- `get_image_bytes(image_ref: str) -> bytes` returns raw PNG bytes from
  the absolute filesystem path stored as `image_ref`.
- Unknown `image_ref` raises `DataError` (a `HarnessError` subclass), as
  required by the docstring on lines 26–32.

The adapter introduces additional **public surface beyond the port** (the
`NIHDatasetConfig` constructor argument), which is permissible per
ARCHITECTURE.md §8.8 ("adapters are not part of the public surface;
consumers wire them via the factories"). The configuration object is
visible only to whatever factory in `harness/composition/factories.py`
constructs the adapter.

Every contract test in `tests/harness/contract/test_dataset_contract.py`
must pass against `TestNIHDatasetContract` without modification to the
port, the abstract contract class, or the `Dataset` / `Sample` domain
types. If any of those need to change, the right answer is a spec PR
modifying ARCHITECTURE.md §3 / §4.1 *and* this document, not an
adapter-side workaround.

End of spec.
