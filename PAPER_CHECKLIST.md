# Paper Checklist — CXR ML, JMIR AI / MIDL target

Tracking doc for the path from current state → published paper. Check items as you go; add notes inline.

**Budget:** $0. Free venues only, free compute only, no APCs, no paid IRB.
**Target venues (in submission order):** arXiv preprint → MIDL workshop (PMLR, free) → ML4H @ NeurIPS (free) → TMLR (peer-reviewed, OpenReview-hosted, no APC) → MICCAI workshops.
**Honest macro-F1 trajectory:** 0.097 (current rule-based) → 0.30+ (with pretrained backbone) → publication-grade.

---

## ✅ Already Done

- [x] Threshold tuning study against NIH labels (macro-F1 0.047 → 0.097) — `analyzer/`
- [x] Calibrated GBT pipeline replacing logistic regression (commit `5e89b8b`)
- [x] Publication-grade experiment harness with ports & adapters (PR #2 **merged** as `965086d`)
  - 9 ports, 9 fakes, 4 sklearn real adapters, filesystem store, composition root
  - 267 tests passing, ruff + mypy strict clean
  - `ARCHITECTURE.md` + §13 v1 surface documented
- [x] Step 3.5 — Feature Cache + Ablation Runner (PR #8 **merged** as `693460f`)
  - `CachedBackbone` adapter (`harness/adapters/fs/cached_backbone.py`) — content-addressable feature cache wrapping any `BackbonePort`
  - `harness/scripts/run_ablation.py` CLI — runs the publication pipeline across a comma-separated seed grid, sharing the cache; writes `comparison.csv`
  - Ablation runner integration test (`tests/harness/integration/test_ablation_runner_smoke.py`) including byte-identical determinism contract across cache miss vs cache hit
  - `ARCHITECTURE.md` §13.7 documents the v1 surface

---

## 🔴 Critical Path (do these in order)

### 1. NIH Dataset Adapter (~2–3 days)
- [x] Read `Data_Entry_2017_v2020.csv` schema (already on disk)
- [x] Spec the multi-label one-hot encoding (14 NIH labels)
- [x] Patient-level metadata extraction (Patient ID column → `patient_id` field)
- [x] TDD: write contract test against `DatasetPort` with a 100-row fixture
- [x] Implement `harness/adapters/fs/nih_dataset.py` against `DatasetPort`
- [x] Image loader: JPG/PNG → numpy float32, resized to backbone input size, normalized
- [ ] Caching layer (memmap or LMDB) so re-runs don't re-decode JPEGs
- [x] Smoke test: load 1k samples, verify shapes + label distribution
- [ ] Full-dataset gate behind `--full` flag so unit tests stay fast

### 2. Torch Backbone Adapter (~2–3 days)
- [x] Add `torch` + `torchvision` to actual install (`pip install -e ".[experiment]"`)
- [x] TDD: contract test against `BackbonePort` using fake images
- [x] Implement `harness/adapters/torch/backbone.py`:
  - [x] `TorchVisionResNet50Backbone` (ImageNet weights from torchvision)
  - [x] `TorchVisionDenseNet121Backbone` (ImageNet weights)
  - [ ] `RadImageNetResNet50Backbone` (RadImageNet weights — verify CC BY 4.0 attribution)
- [x] **Apple Silicon MPS support** — use `torch.device("mps")` if available; works on this Mac, costs $0
- [x] GPU-aware batching: try `mps` → `cuda` → `cpu`, fallback chain
- [x] Deterministic mode: `torch.manual_seed`, `torch.backends.cudnn.deterministic=True`
- [x] Eval-only (`model.eval()` + `torch.no_grad()`); no training in v1
- [x] Mark torch tests with `@pytest.mark.torch`; default test run skips them
- [x] Add `torch` to mypy `ignore_missing_imports` (already done in pyproject)

### 3. First Real Run (~1–2 days)
- [x] Wire NIH + ResNet50 (ImageNet) + existing sklearn head/calibrator/threshold via `composition/factories.py` → add `build_publication_runner_v1(seed)` factory
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
- [ ] **arXiv preprint first** — citable prior art before journal submission ($0)
- [ ] Submission targets (free only, in priority order):
  - [ ] arXiv preprint
  - [ ] MIDL workshop (PMLR proceedings, no APC, accepts indie authors)
  - [ ] ML4H @ NeurIPS workshop (free submission, OpenReview-hosted)
  - [ ] TMLR (Transactions on Machine Learning Research — peer-reviewed, no APC, growing reputation)
  - [ ] MICCAI workshops (specific workshops are free; main conference is paid, skip)
- [ ] **Required ethics statement (free path):** "This study used publicly available, de-identified data from the NIH ChestX-ray14 dataset (Wang et al., 2017). Per the dataset terms of use and consistent with US Common Rule §46.104(d)(4), secondary analysis of publicly available, de-identified data does not constitute human subjects research and was therefore exempt from IRB review."
- [ ] Code + frozen weights URL in paper (link to GitHub repo + release tag)
- [ ] Model card + data card published alongside

---

## 🟡 Parallel Tracks (start now, don't wait)

### A. PR Review
- [x] PR #2 reviewed and merged into `main` (`965086d`)
- [x] PR #4 reviewed and merged into `main` (`fc357e3`) — NIH ChestX-ray14 dataset adapter (Step 1)
- [x] PR #6 reviewed and merged into `main` (`aa7c357`) — TorchVision backbone adapter ResNet50 + DenseNet121 (Step 2)
- [x] PR #7 reviewed and merged into `main` (`60b4274`) — publication runner v1 + sklearn GBT head (Step 3 wiring)
- [x] PR #8 reviewed and merged into `main` (`558ff17`) — feature cache + ablation runner (Step 3.5)
- [ ] Tag release `v0.5.0-harness`

### B. IRB — Self-Attestation Path ($0)
The $300 commercial IRB letter (WCG/Advarra) is **not required** for free venues:
- arXiv has no IRB requirement.
- MIDL, ML4H, TMLR, MICCAI workshops do not require an exemption letter for secondary analysis of public de-identified data — a self-attestation in the ethics statement is accepted.
- Strategy:
  - [ ] Use the ethics-statement language in §5 above (cites Common Rule §46.104(d)(4)).
  - [ ] If a reviewer ever pushes back, **first** point to the dataset's own terms ("free for research and educational purposes") and the standard university IRB determinations (UW, UConn, Berkeley) that classify this work as not-human-subjects-research.
  - [ ] Backup option (still free): some universities will issue a courtesy "not human subjects" determination for unaffiliated researchers using their public datasets — ask via the dataset provider's contact email.
  - [ ] Only spend money on a commercial IRB if a target journal explicitly demands a letter AFTER initial submission. Don't pre-pay.

### C. PhysioNet Credentialing — skip
- [x] Skip for v1 paper. NIH alone is enough; credentialing is free but takes 1–2 weeks and needs a reference contact.

### D. Reading List — papers to cite
- [ ] Cohen et al. 2022, *TorchXRayVision* (MIDL/PMLR) — your primary baseline
- [ ] Rajpurkar et al. 2017, *CheXNet* (arXiv 1711.05225) — foundational
- [ ] Strick, Garcia, Huang 2025, *Reproducing CheXNet* (arXiv 2505.06646) — recent solo-author work
- [ ] Wang et al. 2017, *NIH ChestX-ray8/14* (CVPR) — dataset paper
- [ ] Sechidis et al. 2011, *Iterative Stratification for Multi-Label Data* — for the splitter
- [ ] Fan & Lin, *A Study on Threshold Selection for Multi-label Classification* (NTU) — for SCut threshold work
- [ ] CXR-LT 2024 MICCAI Challenge overview (arXiv 2506.07984) — recent benchmark

### E. Free Compute Plan ($0)
ResNet50 inference on ~112k NIH images is feature-extraction, one-pass-and-cache. Doable for free:
- [ ] **Local first:** Apple Silicon MPS via `torch.device("mps")`. M-series Macs run ResNet inference at usable throughput; ~112k images at ~50ms/image ≈ 1.5 hrs single-pass. Cache features to disk.
- [ ] **Kaggle Notebooks** (free GPU: T4 or P100, 30 hrs/week). Best free GPU tier; upload a small image subset or stream from Kaggle's NIH mirror.
- [ ] **Google Colab free tier** (T4, ~12 hr sessions, can disconnect). Good fallback.
- [ ] **Lightning AI Studios** (free tier with periodic GPU credits).
- [ ] Cache extracted features as `.npy` files; never re-run the backbone unless backbone changes.
- [ ] All ablations downstream of feature extraction (head, calibrator, threshold) run on CPU in seconds.

### F. Legal / Licensing Hygiene
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
- **2026-05-01:** Harness PR #2 merged. Tracking checklist updated to $0 budget: dropped paid IRB letter (use self-attestation citing Common Rule §46.104(d)(4)), dropped APC journals (JMIR AI / Scientific Reports / PLOS One), targeting free venues only (arXiv → MIDL → ML4H → TMLR → MICCAI workshops). Compute plan is local MPS + Kaggle/Colab free tiers.
- **2026-05-01:** Added a feature cache (Step 3.5, PR #8). Ablation runs re-extract the same features for every variant; ResNet50 forward is the long pole at ~50ms/image, which dominates the wall clock. `CachedBackbone` wraps the inner backbone before the decoding/preprocess wrapper so cache keys are content-addressable on raw image bytes + backbone identity, and downstream ablations (head, calibrator, threshold) reuse cached features across the seed grid in seconds.
- **2026-05-01:** Pilot-1 (subset run on 4,999 NIH samples via `harness/scripts/run_pilot.py`) produced macro-F1 = 0.109 / macro-AUROC = 0.643 — barely above the 0.097 rule-based baseline. Flagging for diagnostic on a larger slice (likely the threshold + calibration co-fit on a tiny val fold, or class-imbalance behavior at this sample size) before committing to a full ~112k run. Step 3 boxes (pilot/full-run/gate/TXRV-baseline) intentionally left unchecked pending that diagnostic.
