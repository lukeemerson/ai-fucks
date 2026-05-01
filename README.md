# NIH ChestX-ray14 Heuristic Analyzer

A lightweight chest X-ray demo that uses hand-built grayscale heuristics to surface likely findings from NIH ChestX-ray14 images and present them in a browser dashboard.

This project is intentionally simple:

- no deep learning model weights
- no end-to-end image model
- no claim of clinical validity

The point is to show a clear, explainable prototype that maps image-derived features to findings and high-level urgency tiers, complete with differential diagnosis reasoning and probability-based confidence levels.

## What It Does

For each PNG image, the analyzer extracts 14 classical image features:

| Metric | What it captures |
|---|---|
| `ctr` | Cardiothoracic ratio estimate |
| `ptx_left/right_mean` | Peripheral lucency (mean intensity) |
| `ptx_left/right_std` | Peripheral lucency (texture variance) |
| `ptx_left/right_edge_density` | Vascular marking density via Sobel magnitude |
| `basal_opacity` | Lower-zone opacification |
| `bilateral_haze` | Mid-zone bilateral haziness |
| `diaphragm_pos` | Diaphragm dome position (relative row) |
| `diaphragm_flatness` | Std of dome position across 5 column samples |
| `focal_variance` | Max local variance — focal opacity signature |
| `horiz_band` | Full-lung horizontal Sobel response |
| `lower_horiz_band` | Bibasal Sobel response (rows 55–80%) |

Those metrics are fed through per-finding logistic regression calibrators to produce probabilities for:

| Finding | Tier |
|---|---|
| Pneumothorax | 1 — MUST recognize |
| Pulmonary Edema | 1 — MUST recognize |
| Cardiomegaly | 2 — Should recognize |
| Pleural Effusion | 2 — Should recognize |
| Possible Consolidation / Pneumonia | 2 — Should recognize |
| Atelectasis | 2 — Should recognize |
| Emphysema / Hyperinflation | 3 — Should know |
| Focal Opacity / Nodule | 3 — Should know |

Detections are returned as a `ddx` array sorted by (tier, probability), with confidence labels (high/medium/low) and ranked differential diagnosis considerations per finding.

## Project Structure

```text
analyzer/
  __init__.py          Package marker
  dashboard.html       Single source of truth for the dashboard HTML
  features.py          Classical image feature extraction (14 metrics)
  m4_findings.py       Finding metadata, tier labels, and tier badges
  predict.py           DDx builder, confidence labels, fallback threshold rules
  profile.py           Persisted profile loading, validation, and scoring
  train.py             Train per-finding logistic regression calibrators
  evaluate.py          Confusion matrix + Youden's-J threshold suggestions
  main.py              Batch analyzer entrypoint
fixtures/
  seed_report.json     Small sample report for dashboard demo without images
models/                (gitignored) saved calibration profiles + split reports
results/               (gitignored) populated by main.py or server.py at startup
server.py              Local static server with bootstrap + --port flag
tests/                 pytest suite — 35 tests covering features, evaluate, train
pyproject.toml         Package + dep pins; console scripts below
```

## Requirements

- Python 3.12+
- NIH ChestX-ray14 images at `db-test_images/images/` (only needed for analyzing your own images — the seed fixture lets you demo the dashboard without one)

Install in editable mode:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the Analyzer

Analyze a small batch with the fallback threshold rules:

```bash
python -m analyzer.main --n 10
```

Analyze with a saved calibration profile (recommended):

```bash
python -m analyzer.main --profile models/cxr_profile.json --n 10
python -m analyzer.main --profile models/cxr_profile.json --image /path/to/xray.png
```

Append a larger batch to the existing report:

```bash
python -m analyzer.main --offset 50 --n 200 --append
```

## Viewing the Dashboard

Start the local server (bootstraps `results/` from `analyzer/dashboard.html` + `fixtures/seed_report.json` if empty):

```bash
python3 server.py            # auto-opens http://localhost:8080/dashboard.html
python3 server.py --port 9090 --no-open
```

The dashboard shows sortable findings, tier coloring, probability/confidence badges, and per-finding DDx reasoning.

## Training a Calibration Profile

The primary detection path is a persisted calibration profile — a per-finding logistic regression trained on extracted metrics from labeled images.

1. Extract metrics from a labeled batch (creates `results/report.json`):

```bash
python -m analyzer.main --n 1000
```

2. Train the profile:

```bash
python -m analyzer.train \
  --labels Data_Entry_2017_v2020.csv \
  --report results/report.json \
  --out-profile models/cxr_profile.json
```

This writes:
- `models/cxr_profile.json` — saved per-finding calibration profile
- `models/cxr_profile.train_report.json` — predictions on the train split
- `models/cxr_profile.val_report.json` — predictions on the validation split
- `models/cxr_profile.test_report.json` — predictions on the held-out test split

The split is 70/15/15 and is deterministic (SHA-1 hash of the image filename).

## Evaluating Recognition Quality

Drop `Data_Entry_2017_v2020.csv` (NIH ChestX-ray14 labels) into the project root and run:

```bash
# Evaluate threshold fallback on main report
python -m analyzer.evaluate --labels Data_Entry_2017_v2020.csv

# Evaluate the held-out test split from a trained profile
python -m analyzer.evaluate \
  --labels Data_Entry_2017_v2020.csv \
  --report models/cxr_profile.test_report.json

# Print Youden's-J optimal thresholds for single-metric findings
python -m analyzer.evaluate --labels Data_Entry_2017_v2020.csv --suggest
```

The evaluator prints TP/FP/FN/TN, precision, recall, and F1 per finding.

## Strategy for Refinement

Treat this as a feature-engineering plus calibration loop:

1. Regenerate or extend `results/report.json` on a labeled image set
2. Train a profile with `analyzer.train`
3. Evaluate the held-out test report with `analyzer.evaluate`
4. Inspect weak findings and their FP/FN breakdown
5. Improve the underlying feature functions in `analyzer/features.py`
6. Retrain and compare held-out metrics

Some practical rules:

- if a finding is weak, suspect the feature definition before the classifier
- judge progress from held-out evaluation, not from the dashboard alone
- threshold ceilings in `train.py:THRESHOLD_CEILINGS` prevent val-F1 from over-suppressing rare findings
- use threshold fallback mode only for quick demos without a saved profile

## Tests

```bash
pytest
```

The suite covers: feature extractors (crafted array injection), confusion matrix logic, Youden's-J selection, training/split correctness, DDx record format, and profile loading.

When you change feature extraction or the report format, regenerate the seed fixture:

```bash
python -m analyzer.main --n 25 && cp results/report.json fixtures/seed_report.json
```

## Sharing Notes

- the seed fixture in `fixtures/seed_report.json` makes the dashboard load on a fresh clone — keep it ≤ 25 KB
- explain that this is a heuristic prototype, not a diagnostic tool
- keep the NIH image source separate unless you have the right to redistribute it

## Caveat

This project is for prototyping and educational discussion only. It should not be used for clinical decision-making.
