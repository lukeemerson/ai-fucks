# Paper Checklist — CXR ML, JMIR AI / MIDL target

Tracking doc for the path from current state → published paper. Check items as you go; add notes inline.

**Budget:** $0. Free venues only, free compute only, no APCs, no paid IRB.
**Target venues (in submission order):** arXiv preprint → MIDL workshop (PMLR, free) → ML4H @ NeurIPS (free) → TMLR (peer-reviewed, OpenReview-hosted, no APC) → MICCAI workshops.
**Honest macro-F1 trajectory:** 0.097 (rule-based) → 0.176 (TXRV-emb + LR + pr_sweep_no_shrink, 4999 slice) → ~0.20+ expected at full data (workshop-grade).

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

### Step 3.6: Ablation Matrix (this session, 2026-05-02)

Five backbone × head ablation rows on the same 4999-sample slice:

| backbone               | head            | macro_AUROC | macro_F1 | notes |
|------------------------|-----------------|-------------|----------|-------|
| ImageNet ResNet50      | HGBT            | 0.643       | 0.109    | pilot-1 baseline |
| ImageNet DenseNet121   | HGBT            | 0.538       | 0.073    | architecture matters within ImageNet |
| TXRV DenseNet121       | end-to-end      | 0.750       | n/a      | leakage ceiling |
| TXRV DenseNet121       | HGBT (frozen)   | 0.697       | 0.094    | CXR pretraining + frozen feat |
| TXRV DenseNet121       | LR + balanced   | 0.701       | 0.157    | head choice doubles macro-F1 |

Threshold-strategy ablation (TXRV-emb + LR, four strategies):

| threshold strategy        | macro_F1 | notes |
|---------------------------|----------|-------|
| PR-sweep + 0.5 shrinkage  | 0.157    | current default |
| Fixed 0.5 per class       | 0.037    | naive baseline |
| Youden's J                | 0.161    | only strategy that finds Hernia signal |
| PR-sweep no shrinkage     | 0.176    | best; +12% relative over baseline |

Hernia sensitivity (drop the N=9 class): rank order unchanged across all four ablation runs. TXRV-emb advantage over pilot-1 *widens* from +0.054 to +0.080 AUROC.

Key finding: at the 4999-sample slice, AUROC ranking quality (~0.70) is the binding constraint, not threshold placement. The 0.20 macro-F1 reporting gate is not clearable at this slice size. Full-data run is the unblocking experiment.

PR #10 (LR head adapter, **merged**): logistic regression head with `class_weight='balanced'` rescues 5 previously-zero classes (Mass, Consolidation, Edema, Fibrosis, Pleural_Thickening). Becomes the v1 paper's recommended head.

PR #11 (TXRV backbone adapter, **in-flight**): wraps `torchxrayvision`'s `densenet121-res224-nih` weights as a `BackbonePort` adapter; unlocks the CXR-pretraining ablation cell properly.

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
  - [ ] `RadImageNetResNet50Backbone` (RadImageNet weights — verify CC BY 4.0 attribution; deferred — TXRV is the better CXR-pretrained backbone)
- [x] **Apple Silicon MPS support** — use `torch.device("mps")` if available; works on this Mac, costs $0
- [x] GPU-aware batching: try `mps` → `cuda` → `cpu`, fallback chain
- [x] Deterministic mode: `torch.manual_seed`, `torch.backends.cudnn.deterministic=True`
- [x] Eval-only (`model.eval()` + `torch.no_grad()`); no training in v1
- [x] Mark torch tests with `@pytest.mark.torch`; default test run skips them
- [x] Add `torch` to mypy `ignore_missing_imports` (already done in pyproject)

### 3. First Real Run (~1–2 days)
- [x] Wire NIH + ResNet50 (ImageNet) + existing sklearn head/calibrator/threshold via `composition/factories.py` → add `build_publication_runner_v1(seed)` factory
- [x] Pilot run on 5–10k sample slice — sanity check pipeline end-to-end (pilot-1 ran on 4999, see Step 3.6)
- [x] Pilot at 4999 produced macro-F1 = 0.176 (best threshold strategy on TXRV-emb+LR); 0.20 gate **NOT** cleared at this slice size — confirmed by 3-seed variance bound, threshold ablation, and Hernia sensitivity. The binding constraint at this slice is AUROC ranking quality (~0.70), not threshold placement.
- [ ] Full run on ~112k NIH samples — **blocked on full-NIH download**; Kaggle CDN throughput is slow (~300 KiB/s, ~40 hr ETA from launch)
- [ ] Capture artifacts via `FilesystemArtifactStore` to `runs/<timestamp>/`
- [ ] **Gate: macro-F1 ≥ 0.20 on test split.** Expected to clear at full-data scale (TXRV-emb + LR + pr_sweep_no_shrink).
- [ ] Once full data lands, expected macro-AUROC ~0.78–0.82 on TXRV-disjoint held-out split (clean, no leakage).
- [x] Compare to TorchXRayVision NIH-14 baseline (TXRV-e2e: 0.750 AUROC, leakage-biased upper-bound; see Step 3.6)

### 4. Ablations (~3–4 days)

Already done (see Step 3.6 above):
- [x] **Backbone:** ImageNet ResNet50 vs ImageNet DenseNet121 vs TXRV-pretrained DenseNet121 (frozen)
- [x] **Head:** HGBT vs LR + class-balanced (LR doubles macro-F1 at fixed backbone)
- [x] **Threshold strategy:** fixed 0.5 vs PR-sweep + shrinkage vs PR-sweep no shrinkage vs Youden's J (pr_sweep_no_shrink wins at 0.176)
- [x] **Calibration co-fit reviewer-bait check:** flagged; calibrator + threshold trained on same val fold — document in paper or split with extra fold

Remaining:
- [ ] **External validation:** train on NIH, test on VinDr-CXR or PadChest held-out (no retraining). Adapter work needed (~1–2 day agent).
- [ ] **Fairness slice:** per-sex, per-age-band macro-F1 / AUROC. NIH manifest has both columns — recompute on existing predictions.
- [ ] **Bootstrap CI at n=1000** (currently most ablation runs use n=8 — adequate for cycle time, not paper-final).
- [ ] **Calibration reliability diagrams** per class (none vs Platt vs isotonic).
- [ ] **Confidence intervals on the ablation table cells** (currently point estimates only).
- [ ] All runs use the same seed for reproducibility; capture every config in the model card.

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

## 🎯 Two Tracks — Honest Venue Ambition

**Decision needed.** User should pick a track before committing to the next month of work. Defaulting to Track A (workshop-grade, stay $0) until/unless the user explicitly relaxes a constraint.

### Track A — Honest Workshop Paper ($0, current constraints)

Realistic ceiling: **macro-AUROC ~0.78–0.82** on full-NIH-14 held-out, clean (no leakage). Macro-F1 likely clears 0.20 at full scale.

Realistic venues (in priority):
- arXiv preprint (no peer review, $0)
- MIDL workshop (PMLR, free)
- ML4H @ NeurIPS (free, OpenReview)
- TMLR (peer-reviewed, no APC) — borderline; depends on methods novelty
- MICCAI workshops (specific ones are free)

What gets us there from here:
- [ ] Full-NIH download completes
- [ ] Pilot at n=112120 with TXRV-emb + LR head + pr_sweep_no_shrink threshold
- [ ] External validation on VinDr-CXR or PadChest
- [ ] Bootstrap n=1000 CIs
- [ ] Calibration + fairness analyses
- [ ] arXiv-ready draft with model + data cards

What this paper IS:
- Methodologically careful (hexagonal harness, TDD, ablation-driven)
- Honest about scale and the threshold-budget finding
- A real, defensible workshop contribution

What this paper is NOT:
- A SOTA-claim paper
- Competitive with CheXNet, TorchXRayVision, or CXR-LT 2024 winners
- Suitable for top medical-imaging conferences (MICCAI main, Radiology AI)

### Track B — SOTA-Chasing (requires breaking $0 constraint)

To beat published baselines (~0.84 AUROC for CheXNet, 0.86+ for CXR-LT 2024 winners), at least one of the following must be relaxed:

1. **GPU budget.** End-to-end fine-tuning of a backbone on full NIH-14 needs ~10–50 GPU-hours. Free options:
   - Kaggle Notebooks (free T4/P100, 30 hr/week limit)
   - Colab free tier (T4, 12-hr sessions)
   - Lightning AI Studios (free credits)
   Each requires careful resource management; not impossible at $0 but adds 1–2 weeks of operational complexity.
2. **Multi-dataset training.** CheXpert (Stanford) and MIMIC-CXR (MIT/PhysioNet) require credentialing — free, but takes 1–2 weeks for dataset access approval.
3. **Modern architectures.** Vision Transformers (ViT-L/14, EVA-02) outperform DenseNet121 on CXR by ~0.02–0.05 AUROC; require larger compute footprint.
4. **Ensembling.** Averaging 3–5 backbones typically adds +0.01–0.03 AUROC. Compute-cheap once individual models are trained.
5. **Clinical co-author.** Doesn't move the numbers but improves review prospects at clinical venues; recruitment takes weeks.

What this paper could become:
- TMLR or MIDL main proceedings (with strong numbers + methods contribution)
- ML4H main proceedings (numerical bar is competitive)
- Possibly MICCAI workshop main track (e.g., DEEP-CXR)

What this paper is still not:
- Top medical-imaging conference (MICCAI main, Radiology AI) — those expect clinical evaluation
- Replacing CheXNet — that ship sailed in 2017 and the field has moved on

---

## 🟡 Parallel Tracks (start now, don't wait)

### A. PR Review
- [x] PR #2 reviewed and merged into `main` (`965086d`)
- [x] PR #4 reviewed and merged into `main` (`fc357e3`) — NIH ChestX-ray14 dataset adapter (Step 1)
- [x] PR #6 reviewed and merged into `main` (`aa7c357`) — TorchVision backbone adapter ResNet50 + DenseNet121 (Step 2)
- [x] PR #7 reviewed and merged into `main` (`60b4274`) — publication runner v1 + sklearn GBT head (Step 3 wiring)
- [x] PR #8 reviewed and merged into `main` (`558ff17`) — feature cache + ablation runner (Step 3.5)
- [x] PR #10 reviewed and merged into `main` — logistic-regression head adapter (Step 3.6 head ablation)
- [ ] PR #11 — TXRV CXR-pretrained backbone adapter (Step 3.6 backbone ablation, in-flight)
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
- **2026-05-02:** Step 3.6 ablation matrix complete. Five backbone × head rows on the same 4999-slice produced a clear story: (1) ImageNet DenseNet121+HGBT *underperforms* ImageNet ResNet50+HGBT (0.538 vs 0.643 macro-AUROC) — architecture matters within ImageNet weights; (2) TXRV CXR-pretrained DenseNet121 (frozen embeddings) beats ImageNet ResNet50 (~0.70 vs 0.64 AUROC) — domain-pretrained beats ImageNet at this slice; (3) TXRV-e2e is 0.750 AUROC but leakage-biased (TXRV trained on the same NIH corpus our test split is drawn from); (4) head choice (LR + class_weight='balanced' vs HGBT) doubles macro-F1 (0.157 vs 0.094) at fixed backbone — LR rescues 5 previously-zero classes (Mass, Consolidation, Edema, Fibrosis, Pleural_Thickening); (5) threshold-strategy ablation: pr_sweep_no_shrink wins (0.176 macro-F1), beating pr_sweep + 0.5 shrinkage (0.157, current default) and Youden's J (0.161). Hernia sensitivity (drop the N=9 class) leaves rank order unchanged across all four ablation runs and *widens* the TXRV-emb advantage from +0.054 to +0.080 AUROC — robust signal, not driven by Hernia variance. **Headline finding:** at the 4999-slice, AUROC ranking quality (~0.70) is the binding constraint, not threshold placement; the 0.20 macro-F1 reporting gate is not clearable here. Full-data run (~112k) is the unblocking experiment, currently blocked on slow Kaggle CDN throughput.
- **2026-05-02:** Added Two-Tracks framing (§Two Tracks). Track A is the honest $0 workshop path (target macro-AUROC ~0.78–0.82 at full data; arXiv → MIDL → ML4H → TMLR → MICCAI workshops). Track B is SOTA-chasing and requires relaxing at least one $0 constraint (free GPU credits, multi-dataset credentialing, modern architectures, ensembling, or a clinical co-author). User decision required before committing the next month of work; defaulting to Track A.
