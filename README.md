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
- consolidation
- atelectasis
- emphysema
- focal opacity

Each finding is tagged with a high-level urgency tier:

- Tier 1: must recognize
- Tier 2: should recognize
- Tier 3: should know about

The results are written to `results/report.json` and rendered in a browser dashboard.

## Project Structure

```text
analyzer/
  dashboard.html       Dashboard template copied into results/
  features.py          Classical image feature extraction
  m4_findings.py       Finding metadata and tier labels
  main.py              Batch analyzer entrypoint
results/
  dashboard.html       Served demo dashboard
  report.json          Generated analysis output
server.py              Local static server for the results dashboard
```

## Requirements

- Python 3.12 recommended
- package dependencies from `analyzer/requirements.txt`
- a local NIH ChestX-ray14 image folder at `db-test_images/images/`

Install dependencies into a virtual environment:

```bash
python3.12 -m venv analyzer/.venv
source analyzer/.venv/bin/activate
python -m pip install -r analyzer/requirements.txt
```

## Running The Analyzer

Analyze a small batch:

```bash
source analyzer/.venv/bin/activate
python analyzer/main.py --n 10
```

Append a larger batch to the existing report:

```bash
source analyzer/.venv/bin/activate
python analyzer/main.py --offset 50 --n 200 --append
```

What the flags mean:

- `--n`: number of images to process
- `--offset`: how many alphabetically sorted images to skip
- `--append`: merge into the existing `results/report.json` without duplicating filenames

## Viewing The Dashboard

Start the local server:

```bash
python3 server.py
```

Then open:

```text
http://localhost:8080/dashboard.html
```

## Current Demo Snapshot

At the time of this README, the included generated report contains:

- 250 analyzed images
- a dashboard optimized for browsing larger batches
- summary cards, finding-frequency charts, tier distribution, and a searchable case browser

## Sharing Notes

If you want to publish or demo this cleanly:

- keep the generated `results/` files so people can open the dashboard immediately
- explain that this is a heuristic prototype, not a diagnostic tool
- keep the NIH image source separate unless you have the right to redistribute it

A good one-line description is:

> A chest X-ray triage demo that uses explainable pixel heuristics to flag likely findings and map them to high-level urgency tiers.

## Caveat

This project is for prototyping and educational discussion only. It should not be used for clinical decision-making.
