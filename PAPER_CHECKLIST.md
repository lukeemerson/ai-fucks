# Paper Checklist — CXR ML, JMIR AI / MIDL target

Tracking doc for the path from current state → published paper. Check items as you go; add notes inline.

**Target venues (in submission order):** arXiv preprint → MIDL or ML4H workshop → JMIR AI / Scientific Reports.
**Honest macro-F1 trajectory:** 0.097 (current rule-based) → 0.30+ (with pretrained backbone) → publication-grade.

---

## ✅ Already Done

- [x] Threshold tuning study against NIH labels (macro-F1 0.047 → 0.097) — `analyzer/`
- [x] Calibrated GBT pipeline replacing logistic regression (commit `5e89b8b`)
- [x] Publication-grade experiment harness with ports & adapters (PR #2, branch `pate/harness-publication-ready`)
  - 9 ports, 9 fakes, 4 sklearn real adapters, filesystem store, composition root
  - 267 tests passing, ruff + mypy strict clean
  - `ARCHITECTURE.md` + §13 v1 surface documented

---

## 🔴 Critical Path (do these in order)

### 1. NIH Dataset Adapter (~2–3 days)
- [ ] Read `Data_Entry_2017_v2020.csv` schema (already on disk)
- [ ] Spec the multi-label one-hot encoding (14 NIH labels)
- [ ] Patient-level metadata extraction (Patient ID column → `patient_id` field)
- [ ] TDD: write contract test against `DatasetPort` with a 100-row fixture
- [ ] Implement `harness/adapters/fs/nih_dataset.py` against `DatasetPort`
- [ ] Image loader: JPG/PNG → numpy float32, resized to backbone input size, normalized
- [ ] Caching layer (memmap or LMDB) so re-runs don't re-decode JPEGs
- [ ] Smoke test: load 1k samples, verify shapes + label distribution
- [ ] Full-dataset gate behind `--full` flag so unit tests stay fast

### 2. Torch Backbone Adapter (~2–3 days)
- [ ] Add `torch` + `torchvision` to actual install (`pip install -e ".[experiment]"`)
- [ ] TDD: contract test against `BackbonePort` using fake images
- [ ] Implement `harness/adapters/torch/backbone.py`:
  - [ ] `TorchVisionResNet50Backbone` (ImageNet weights from torchvision)
  - [ ] `TorchVisionDenseNet121Backbone` (ImageNet weights)
  - [ ] `RadImageNetResNet50Backbone` (RadImageNet weights — verify CC BY 4.0 attribution)
- [ ] GPU-aware batching with `torch.cuda.is_available()` fallback to CPU
- [ ] Deterministic mode: `torch.manual_seed`, `torch.backends.cudnn.deterministic=True`
- [ ] Eval-only (`model.eval()` + `torch.no_grad()`); no training in v1
- [ ] Mark torch tests with `@pytest.mark.torch`; default test run skips them
- [ ] Add `torch` to mypy `ignore_missing_imports` (already done in pyproject)

### 3. First Real Run (~1–2 days)
- [ ] Wire NIH + ResNet50 (ImageNet) + existing sklearn head/calibrator/threshold via `composition/factories.py` → add `build_publication_runner_v1(seed)` factory
- [ ] Pilot run on 5–10k sample slice — sanity check pipeline end-to-end
- [ ] Full run on ~112k NIH samples (CPU may be feasible since features are extracted once and cached)
- [ ] Capture artifacts via `FilesystemArtifactStore` to `runs/<timestamp>/`
- [ ] **Gate: macro-F1 ≥ 0.20 on test split.** If below, debug before continuing.
- [ ] Compare to TorchXRayVision NIH-14 baseline (use TXRV directly as a separate run, not as a dep)

### 4. Ablations (~3–4 days)
Each ablation is a separate run with one variable swapped:
- [ ] **Backbone:** ImageNet ResNet50 vs RadImageNet ResNet50 vs DenseNet121 vs scratch
- [ ] **Loss / head:** BCE vs class-weighted BCE vs focal loss
- [ ] **Threshold strategy:** fixed 0.5 vs per-class PR-sweep vs PR-sweep + shrinkage (your novel contribution)
- [ ] **Calibration:** none vs Platt vs isotonic
- [ ] **External validation:** train on NIH, test on VinDr-CXR or PadChest held-out (no retraining)
- [ ] **Fairness slice:** per-sex, per-age-band macro-F1
- [ ] All runs use the same seed for reproducibility; capture every config in the model card

### 5. Paper Draft (~2–3 weeks)
- [ ] Outline structure (Introduction / Related Work / Method / Experiments / Discussion)
- [ ] Method section is the harness architecture + threshold-tuning algorithm
- [ ] Experiments section is the ablation table + external validation table
- [ ] Per-label AUROC + 95% CI via bootstrap (already in `BootstrapMetrics`)
- [ ] Calibration reliability diagrams (per-class)
- [ ] At least 5 papers cited as direct comparisons (see Reading List below)
- [ ] **arXiv preprint first** — citable prior art before journal submission
- [ ] Submission target picks (in priority order): MIDL workshop, ML4H, JMIR AI, Scientific Reports
- [ ] Required ethics statement: "This study used publicly available, de-identified data and was therefore exempt from IRB review under [exemption letter from §B below]."
- [ ] Code + frozen weights URL in paper (link to GitHub repo + release tag)
- [ ] Model card + data card published alongside

---

## 🟡 Parallel Tracks (start now, don't wait)

### A. PR Review
- [ ] PR #2 reviewed and merged into `main`
- [ ] Tag release `v0.5.0-harness` once merged

### B. IRB Exemption Letter (~$300, half day)
- [ ] Pick provider (WCG or Advarra)
- [ ] Submit exemption application: "secondary analysis of NIH ChestX-ray14 (publicly available, de-identified)"
- [ ] Receive letter — keep PDF in `harness/docs/IRB/`
- [ ] Reference number ready for journal submission ethics statement

### C. PhysioNet Credentialing — only if MIMIC-CXR gets added later
- [ ] Skip for v1 paper (NIH alone is enough)
- [ ] If pursuing v1.1: register on CITI under "MIT Affiliates"
- [ ] Complete "Data or Specimens Only Research" course (~2 hrs, free)
- [ ] Find a reference contact (this is the real bottleneck)
- [ ] Submit credentialing app on physionet.org

### D. Reading List — papers to cite
- [ ] Cohen et al. 2022, *TorchXRayVision* (MIDL/PMLR) — your primary baseline
- [ ] Rajpurkar et al. 2017, *CheXNet* (arXiv 1711.05225) — foundational
- [ ] Strick, Garcia, Huang 2025, *Reproducing CheXNet* (arXiv 2505.06646) — recent solo-author work
- [ ] Wang et al. 2017, *NIH ChestX-ray8/14* (CVPR) — dataset paper
- [ ] Sechidis et al. 2011, *Iterative Stratification for Multi-Label Data* — for the splitter
- [ ] Fan & Lin, *A Study on Threshold Selection for Multi-label Classification* (NTU) — for SCut threshold work
- [ ] CXR-LT 2024 MICCAI Challenge overview (arXiv 2506.07984) — recent benchmark

### E. Legal / Licensing Hygiene
- [ ] Confirm RadImageNet attribution requirements before using their weights
- [ ] Document NIH attribution in the paper (link to NIH ChestX-ray14 download + cite Wang 2017)
- [ ] Decide commercial future: if yes, retrain final weights on NIH + RadImageNet only (skip MIMIC/CheXpert backbones)
- [ ] License repo as MIT or Apache-2.0; include `LICENSE` file
- [ ] Each released artifact gets a model card with training data + license disclosure

---

## 🟢 Submission Checklist (final pre-submit pass)

- [ ] arXiv preprint posted (with code repo URL)
- [ ] GitHub release tagged with frozen weights as a release asset (or HuggingFace model upload)
- [ ] Model card + data card complete and published
- [ ] Reproducibility verified: someone other than you runs the harness from scratch and reproduces the headline number within tolerance
- [ ] All tables in the paper produced by `harness/composition` runs (no manual edits)
- [ ] Per-class AUROC with 95% CI for every label in the paper
- [ ] External validation table (NIH→VinDr or NIH→PadChest)
- [ ] Calibration diagrams included
- [ ] Fairness/subgroup analysis included
- [ ] IRB exemption letter referenced in ethics statement
- [ ] Cover letter drafted (for journal submission)
- [ ] No reviewer-bait: thresholds learned via OOF, calibration honest, no test-set tuning

---

## 🚫 Out of Scope for v1 Paper

- MIMIC-CXR / CheXpert datasets (commercial restrictions, credentialing friction)
- Open-i (per-image licensing chaos)
- LLM-generated radiology reports
- Real-time / clinical deployment
- Multi-modal (radiology report + image)

---

## Notes / Decisions Log

_(append-only; date each entry)_

- **2026-05-01:** Decided to use NIH ChestX-ray14 only for v1 paper. Commercial-safe + no credentialing needed. RadImageNet (CC BY 4.0) acceptable for backbone.
- **2026-05-01:** Decided against torch in `harness/` core; behind `[experiment]` extra so test suite stays fast.
- **2026-05-01:** Per-class learned thresholds via OOF PR-sweep + shrinkage to global pooled anchor + clamps is the methodological contribution.
