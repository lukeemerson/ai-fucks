# Synthetic NIH-14 Fixture

A tiny synthetic dataset that mirrors the NIH ChestX-ray14 schema. Tests use
this fixture so they never have to touch the real ~112k-row
`Data_Entry_2017_v2020.csv` or its image archive.

## File layout

```
tests/harness/fixtures/nih/
├── Data_Entry_synthetic.csv   # 16 rows + header, NIH-14 schema verbatim
├── README.md                  # this file
└── images/                    # 16 PNGs, 8x8 grayscale, names match CSV
    ├── 00000001_000.png
    ├── 00000001_001.png
    ├── ...
    └── 00000005_001.png
```

The CSV header replicates the real file byte-for-byte:

```
Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]
```

The bracketed columns `OriginalImage[Width,Height]` and
`OriginalImagePixelSpacing[x,y]` are each split into two real CSV fields
(width/height, x/y). Each data row therefore has 11 fields.

## Sample / patient / label distribution

5 patients, 16 rows total.

| Patient ID | Rows | Finding Labels                                                                               |
|-----------:|-----:|----------------------------------------------------------------------------------------------|
| 1          | 3    | Cardiomegaly; Effusion; Pneumothorax                                                         |
| 2          | 2    | Cardiomegaly\|Edema; No Finding                                                              |
| 3          | 4    | Mass\|Nodule\|Pneumonia; Atelectasis\|Emphysema; Infiltration; Consolidation\|Fibrosis       |
| 4          | 5    | No Finding x3; Hernia; Pleural_Thickening                                                    |
| 5          | 2    | No Finding x2                                                                                |

Per-label occurrence counts across the 16 rows:

| Label              | Count |
|--------------------|------:|
| Atelectasis        | 1     |
| Cardiomegaly       | 2     |
| Effusion           | 1     |
| Infiltration       | 1     |
| Mass               | 1     |
| Nodule             | 1     |
| Pneumonia          | 1     |
| Pneumothorax       | 1     |
| Consolidation      | 1     |
| Edema              | 1     |
| Emphysema          | 1     |
| Fibrosis           | 1     |
| Pleural_Thickening | 1     |
| Hernia             | 1     |
| No Finding         | 6     |

All 14 NIH disease labels appear at least once. The mix exercises:

- single-label rows (Patient 1)
- multi-label rows (Patient 2: 2-way; Patient 3: 2-way and 3-way)
- "No Finding" rows (Patients 2, 4, 5)
- rare-label paths (Patient 4 carries the only Hernia and Pleural_Thickening rows)
- non-zero `Follow-up #` (Patient 4 uses 10..14)
- both genders and both view positions
- `Patient Age` integers in 30..75
- `OriginalImage[Width,Height]` in the realistic 2500..3000 range
- `OriginalImagePixelSpacing[x,y]` of 0.143 (the real file's modal value)

## Image filenames and contents

Filenames follow the NIH convention
`<patient_id zero-padded to 8>_<follow_up zero-padded to 3>.png`, e.g.
`00000001_000.png`. Each PNG matches a CSV row's `Image Index` field exactly.

Each PNG is **8x8 pixels, grayscale (PIL mode `"L"`), saved as PNG**. Pixel
values are deterministic so regenerating the fixture is byte-stable:

```
pixel(i, j) = (patient_id * 32 + follow_up * 8 + i * 4 + j) % 256
```

with `i` the row (0..7) and `j` the column (0..7). Every image is unique
across the fixture and reproducible from `(patient_id, follow_up)` alone.

## Why this shape

- **Tiny** — total fixture is well under 100 KB; safe to commit.
- **Schema-faithful** — anything that parses the real `Data_Entry_2017_v2020.csv`
  parses this file with the same code path (column count, header, quoting
  conventions all match).
- **Label-complete** — every one of the 14 NIH labels is represented, so
  per-class metric / threshold code never short-circuits on an empty class.
- **Patient-grouped** — multiple rows per patient lets group-aware splitters
  (no patient leakage between train/val/test) be exercised.
- **Deterministic** — both CSV rows and image bytes are fixed; tests can hash
  the fixture if they need a tamper check.

## Regenerating the fixture

The fixture is **static checked-in data**. The test suite must not regenerate
it at runtime. If you need to rebuild it (e.g. to add rows), use a one-shot
script analogous to the original generator:

```python
import csv
from pathlib import Path
import numpy as np
from PIL import Image

FIX = Path("tests/harness/fixtures/nih")
IMG = FIX / "images"
IMG.mkdir(parents=True, exist_ok=True)

HEADER = [
    "Image Index", "Finding Labels", "Follow-up #", "Patient ID",
    "Patient Age", "Patient Gender", "View Position",
    "OriginalImage[Width", "Height]",
    "OriginalImagePixelSpacing[x", "y]",
]

# (patient_id, follow_up, finding_labels, age, gender, view, w, h, sx, sy)
ROWS = [
    (1, 0, "Cardiomegaly",            57, "M", "PA", 2682, 2749, 0.143, 0.143),
    # ... see the table above for the full 16 rows ...
]

def name(pid, fu): return f"{pid:08d}_{fu:03d}.png"

with (FIX / "Data_Entry_synthetic.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    for pid, fu, lbl, *meta in ROWS:
        w.writerow([name(pid, fu), lbl, fu, pid, *meta])

for pid, fu, *_ in ROWS:
    arr = np.fromfunction(
        lambda i, j: (pid * 32 + fu * 8 + i * 4 + j) % 256,
        (8, 8), dtype=np.int64,
    ).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(IMG / name(pid, fu), format="PNG")
```

Then commit the resulting CSV and `images/` directory. The script itself is
not committed; the fixture artifacts are.
