# NIH ChestX-ray14 Heuristic Analyzer

A lightweight chest X-ray demo that uses hand-built grayscale heuristics to surface likely findings from NIH ChestX-ray14 images and present them in a browser dashboard.

This project is intentionally simple:

- no deep learning model weights
- no training pipeline
- no claim of clinical validity

The point is to show a clear, explainable prototype that maps image-derived features to findings and high-level urgency tiers.

## What It Does

For each PNG image, the analyzer extracts a small set of classical image features such as:

- cardiothoracic ratio estimate
- peripheral lucency
- basal opacification
- bilateral haze
- diaphragm position
- focal variance

Those metrics are run through threshold-based heuristics to flag findings like:

- pneumothorax
- pulmonary edema
- cardiomegaly
- pleural effusion
- consolidation (diffuse airspace opacity)
- focal opacity (sharp focal lesion — mutually exclusive with consolidation on `focal_variance`)
- atelectasis
- emphysema

Each finding is tagged with a high-level urgency tier:

- Tier 1: must recognize
- Tier 2: should recognize
- Tier 3: should know about

The results are written to `results/report.json` and rendered in a browser dashboard.

## Project Structure

```text
analyzer/
  __init__.py          Package marker
  dashboard.html       Single source of truth for the dashboard HTML
  features.py          Classical image feature extraction
  m4_findings.py       Finding metadata and tier labels
  main.py              Batch analyzer entrypoint (python -m analyzer.main)
  evaluate.py          Confusion matrix + threshold suggestions vs NIH labels
  tune.py              Joint optimizer for compound-rule cutoffs
fixtures/
  seed_report.json     Small sample report shipped with the repo so a fresh
                       clone can run `python3 server.py` and see a populated
                       dashboard with no images on disk.
results/                (gitignored) populated by main.py or by server.py at
                       startup
server.py              Local static server with bootstrap + --port flag
tests/                 pytest suite (run with `pytest`)
pyproject.toml         Package + dep pins; installs `cxr-analyze`,
                       `cxr-evaluate`, and `cxr-serve` console scripts
```

## Requirements

- Python 3.12 recommended
- a local NIH ChestX-ray14 image folder at `db-test_images/images/` (only needed for analyzing your own images — the seed fixture lets you demo the dashboard without one)

Install in editable mode:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running The Analyzer

Analyze a small batch:

```bash
python -m analyzer.main --n 10
```

Append a larger batch to the existing report:

```bash
python -m analyzer.main --offset 50 --n 200 --append
```

Flags:

- `--n`: number of images to process
- `--offset`: how many alphabetically sorted images to skip
- `--append`: merge into the existing `results/report.json` without duplicating filenames

## Viewing The Dashboard

Start the local server (it will bootstrap `results/` from `analyzer/dashboard.html` + `fixtures/seed_report.json` if empty):

```bash
python3 server.py            # auto-opens http://localhost:8080/dashboard.html
python3 server.py --port 9090 --no-open
```

## Validating Recognition Quality

Without ground truth there is no recognition number. Drop the NIH labels file
(`Data_Entry_2017_v2020.csv`) into the project root and run:

```bash
python -m analyzer.evaluate --labels Data_Entry_2017_v2020.csv
python -m analyzer.evaluate --labels Data_Entry_2017_v2020.csv --suggest
```

The evaluator prints a confusion matrix (TP/FP/FN/TN, precision, recall, F1)
per finding. With `--suggest` it also reports a Youden's-J–optimal threshold
for each single-metric finding.

Compound findings (`pneumothorax`, `consolidation`) need joint optimization
across multiple metrics. The `analyzer.tune` module does that automatically
on an 80/20 train/test split and reports per-rule best cutoffs:

```bash
python -m analyzer.tune --labels Data_Entry_2017_v2020.csv
```

NIH label mapping is documented in `analyzer/evaluate.py:NIH_POSITIVE_LABELS`.

## Tests

```bash
pytest
```

The test suite pins thresholds against the seed fixture, so any change to `THRESHOLDS` that flips a detection on the seed metrics fails loudly. If the change was intentional, regenerate the seed and update the fixture:

```bash
python -m analyzer.main --n 25 && cp results/report.json fixtures/seed_report.json
```

Only do this when you intentionally want to update the pinned thresholds-vs-metrics fixture.

## Sharing Notes

If you want to publish or demo this cleanly:

- the seed fixture in `fixtures/seed_report.json` is what makes the dashboard load on a fresh clone — keep it small (≤25 KB)
- explain that this is a heuristic prototype, not a diagnostic tool
- keep the NIH image source separate unless you have the right to redistribute it

A good one-line description:

> A chest X-ray triage demo that uses explainable pixel heuristics to flag likely findings and map them to high-level urgency tiers.

## Caveat

This project is for prototyping and educational discussion only. It should not be used for clinical decision-making.
