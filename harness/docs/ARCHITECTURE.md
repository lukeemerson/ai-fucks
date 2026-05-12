# Harness Architecture

**Status:** v1 spec, source of truth.
**Audience:** every downstream agent contributing to `harness/`.
**Scope:** publication-grade chest X-ray (CXR) ML experiment harness, intended for a JMIR AI / MIDL workshop submission.
**Discipline:** strict ports & adapters (hexagonal). No exceptions, no shortcuts.

This document defines the *only* architecture the `harness/` module is allowed to follow. Pull requests that violate the rules below will be rejected. Deviations require an updated spec PR landing first.

---

## 1. Goals & Constraints

### 1.1 What `harness/` is

A reproducible, publication-grade experiment runner that orchestrates a CXR multi-label classification pipeline:

1. Load a dataset (image paths + multi-label vectors + patient ids).
2. Build patient-level multi-label stratified train/val/test splits.
3. Extract image features with a frozen backbone.
4. Fit a multi-label classifier head on features.
5. Calibrate the head's per-class probabilities on validation OOF predictions.
6. Tune per-class operating thresholds via OOF PR-sweep with shrinkage and clamps.
7. Compute metrics (macro-F1, per-class AUROC, per-class AUPRC) with bootstrap CIs.
8. Persist artifacts: model card, data card, frozen weights, predictions, threshold set, config hash.

### 1.2 What `harness/` is **not**

- Not a replacement for `analyzer/`. The two are siblings. `analyzer/` stays untouched.
- Not a training framework for backbones. Backbone weights are *frozen* in v1.
- Not a dataset downloader. v1 does not download NIH; tests use the fake dataset adapter.
- Not a serving runtime. Serving lives in `server.py`.

### 1.3 Hard constraints

- Python 3.12.
- ruff (strict rule set) and mypy (`--strict`) must pass.
- pytest with fast unit + contract suites: `pytest tests/harness/unit tests/harness/contract` must complete in under 5 seconds on a laptop and must not import torch.
- No third-party heavy deps in `harness/domain/` or `harness/ports/`. Standard library + `typing` only.
- No torch import outside `harness/adapters/torch/`.
- No filesystem writes outside `harness/adapters/fs/`.
- No `Any`. No bare `# type: ignore`. Justified `# type: ignore[code]  # reason` only.

---

## 2. Module Layout

```
harness/
  __init__.py
  domain/
    __init__.py
    types.py            # dataclasses: Sample, Dataset, Split, ...
    errors.py           # HarnessError hierarchy, no I/O
  ports/
    __init__.py
    dataset.py          # DatasetPort
    splitter.py         # SplitterPort
    backbone.py         # BackbonePort
    head.py             # ClassifierHeadPort
    calibrator.py       # CalibratorPort
    threshold.py        # ThresholdPort
    metrics.py          # MetricsPort
    store.py            # ArtifactStorePort
    randomness.py       # RandomnessPort
  adapters/
    __init__.py
    fakes/
      __init__.py
      dataset.py        # InMemoryFakeDataset
      splitter.py       # DeterministicFakeSplitter
      backbone.py       # HashFakeBackbone
      head.py           # FakeClassifierHead
      calibrator.py     # IdentityFakeCalibrator
      threshold.py      # FixedFakeThreshold
      metrics.py        # FakeMetrics (deterministic fixed report)
      store.py          # InMemoryFakeStore
      randomness.py     # SeededFakeRandomness
    sklearn/
      __init__.py
      head.py           # SklearnLogisticHead, SklearnGBTHead
      calibrator.py     # IsotonicCalibrator, SigmoidCalibrator
      threshold.py      # PrSweepThreshold (OOF PR-sweep + shrinkage + clamps)
      metrics.py        # SklearnMetrics (macro-F1, AUROC, AUPRC, bootstrap CIs)
      splitter.py       # IterativeStratificationSplitter (patient-level)
      randomness.py     # NumpySeededRandomness
    torch/              # gated behind extras; not required for v1 tests
      __init__.py
      backbone.py       # TorchVisionBackbone (deferred)
      dataset.py        # TorchImageFolderDataset (deferred)
    fs/
      __init__.py
      store.py          # LocalFsArtifactStore (writes JSON + npy + .pkl)
  composition/
    __init__.py
    runner.py           # ExperimentRunner.run_experiment(...)
    factories.py        # build_v1_runner_with_fakes(), build_v1_runner_sklearn()
  docs/
    ARCHITECTURE.md     # this document
tests/
  harness/
    unit/
      domain/
        test_types.py
      adapters/
        sklearn/
          test_pr_sweep_threshold.py
          test_isotonic_calibrator.py
          test_sklearn_metrics.py
          test_iterative_splitter.py
        fakes/
          test_in_memory_dataset.py
          test_seeded_randomness.py
        fs/
          test_local_fs_store.py
    contract/
      test_dataset_port_contract.py
      test_splitter_port_contract.py
      test_backbone_port_contract.py
      test_head_port_contract.py
      test_calibrator_port_contract.py
      test_threshold_port_contract.py
      test_metrics_port_contract.py
      test_store_port_contract.py
      test_randomness_port_contract.py
    integration/
      test_runner_golden_path.py
      test_runner_reproducibility.py
      conftest.py
```

### 2.1 Layering rules

- `domain/` may import from stdlib only.
- `ports/` may import from `domain/` and stdlib only.
- `adapters/*/` may import from `domain/`, `ports/`, and the adapter's specific third-party dep. **Never** from another adapter package.
- `composition/` may import from `domain/`, `ports/`, and `adapters/`. It is the only place adapters are constructed.
- Tests may import anything in `harness/`.

Any import that crosses a boundary the wrong way is a bug.

---

## 3. Domain Types

All types live in `harness/domain/types.py`. All are `@dataclass(frozen=True, slots=True)` unless noted. No methods beyond `__post_init__` validation. No I/O. No third-party deps.

Conventions:

- `numpy` arrays are *not* allowed in `domain/`. Probability and label matrices are passed as `tuple[tuple[float, ...], ...]` at the domain boundary, or as opaque `ProbabilityMatrix` / `LabelMatrix` newtypes that adapters can satisfy with numpy under the hood. v1 uses simple nested tuples in domain types; adapters convert.
- IDs are `str`. Indices are `int`. Times are `datetime` (UTC).

### 3.1 `Sample`

```python
@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    patient_id: str
    image_ref: str            # opaque path/URI; adapters resolve
    labels: tuple[int, ...]   # multi-hot, length == len(label_names)
    metadata: Mapping[str, str]  # view position, age band, etc.
```

### 3.2 `Dataset`

```python
@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    label_names: tuple[str, ...]
    samples: tuple[Sample, ...]
```

Invariant: every `sample.labels` has `len == len(label_names)`. Validated in `__post_init__`.

### 3.3 `Split`

```python
@dataclass(frozen=True, slots=True)
class Split:
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    seed: int
```

Invariant: index sets are disjoint, union is a subset of `range(n_samples)`, no patient appears in more than one set (enforced by splitter, asserted in contract tests).

### 3.4 `Probabilities`

```python
@dataclass(frozen=True, slots=True)
class Probabilities:
    sample_ids: tuple[str, ...]            # row order
    label_names: tuple[str, ...]           # column order
    values: tuple[tuple[float, ...], ...]  # shape (n_samples, n_labels), in [0, 1]
```

Invariant: all values in `[0.0, 1.0]`. Row count == `len(sample_ids)`. Column count == `len(label_names)`.

### 3.5 `Predictions`

```python
@dataclass(frozen=True, slots=True)
class Predictions:
    sample_ids: tuple[str, ...]
    label_names: tuple[str, ...]
    values: tuple[tuple[int, ...], ...]    # 0/1
```

### 3.6 `MetricReport`

```python
@dataclass(frozen=True, slots=True)
class MetricInterval:
    point: float
    lower: float
    upper: float

@dataclass(frozen=True, slots=True)
class PerClassMetric:
    label: str
    f1: MetricInterval
    auroc: MetricInterval
    auprc: MetricInterval
    support: int

@dataclass(frozen=True, slots=True)
class MetricReport:
    macro_f1: MetricInterval
    macro_auroc: MetricInterval
    macro_auprc: MetricInterval
    per_class: tuple[PerClassMetric, ...]
    n_bootstrap: int
    seed: int
```

### 3.7 `ThresholdSet`

```python
@dataclass(frozen=True, slots=True)
class ThresholdSet:
    label_names: tuple[str, ...]
    thresholds: tuple[float, ...]   # one per label
    method: str                     # e.g. "pr_sweep+shrink"
    shrinkage: float                # in [0, 1]
    clamp_lo: float
    clamp_hi: float
```

Invariant: `len(thresholds) == len(label_names)`; each threshold in `[clamp_lo, clamp_hi] subseteq [0, 1]`.

### 3.8 `ModelCard`

```python
@dataclass(frozen=True, slots=True)
class ModelCard:
    name: str
    version: str
    created_at: datetime
    backbone: str
    head: str
    calibrator: str
    threshold_method: str
    label_names: tuple[str, ...]
    train_size: int
    val_size: int
    test_size: int
    config_hash: str
    metrics: MetricReport
    notes: str
```

### 3.9 `ExperimentConfig`

```python
@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    n_resamples: int
    confidence: float       # e.g. 0.95
    seed: int

@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    method: str             # "pr_sweep"
    shrinkage: float
    clamp_lo: float
    clamp_hi: float

@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_name: str
    dataset_name: str
    label_names: tuple[str, ...]
    val_fraction: float
    test_fraction: float
    seed: int
    bootstrap: BootstrapConfig
    threshold: ThresholdConfig
    backbone_id: str
    head_id: str
    calibrator_id: str
    artifact_root: str      # opaque to domain; ArtifactStorePort interprets
    notes: str
```

### 3.10 `ExperimentResult`

```python
@dataclass(frozen=True, slots=True)
class ExperimentResult:
    config: ExperimentConfig
    split: Split
    thresholds: ThresholdSet
    val_probabilities: Probabilities
    test_probabilities: Probabilities
    test_predictions: Predictions
    report: MetricReport
    model_card: ModelCard
    artifact_uris: Mapping[str, str]   # logical name -> URI returned by store
```

### 3.11 Errors

`harness/domain/errors.py`:

```python
class HarnessError(Exception): ...
class DomainValidationError(HarnessError): ...
class PortContractError(HarnessError): ...
class AdapterError(HarnessError): ...
```

Adapters raise `AdapterError` (or a subclass) for adapter-specific failures. Contract tests assert that adapter failures funnel through `HarnessError`.

---

## 4. Ports

All ports live in `harness/ports/`. Each is a `typing.Protocol` decorated with `@runtime_checkable`. Method bodies are `...`. No state.

```python
from typing import Protocol, runtime_checkable
```

### 4.1 `DatasetPort`

`harness/ports/dataset.py`

```python
@runtime_checkable
class DatasetPort(Protocol):
    def load(self) -> Dataset: ...
    def get_image_bytes(self, image_ref: str) -> bytes: ...
```

`load()` returns the full `Dataset` (samples + label names). `get_image_bytes` is the only image I/O surface; backbones consume bytes, never paths.

### 4.2 `SplitterPort`

`harness/ports/splitter.py`

```python
@runtime_checkable
class SplitterPort(Protocol):
    def split(
        self,
        dataset: Dataset,
        *,
        val_fraction: float,
        test_fraction: float,
        seed: int,
    ) -> Split: ...
```

Contract: patient-level (no patient leakage), multi-label stratified within the patient constraint.

### 4.3 `BackbonePort`

`harness/ports/backbone.py`

```python
@runtime_checkable
class BackbonePort(Protocol):
    @property
    def feature_dim(self) -> int: ...
    @property
    def identifier(self) -> str: ...
    def extract(self, image_bytes_batch: Sequence[bytes]) -> tuple[tuple[float, ...], ...]: ...
```

`extract` returns a `(batch_size, feature_dim)` matrix as nested tuples. Adapters may use numpy/torch internally and convert at the boundary.

### 4.4 `ClassifierHeadPort`

`harness/ports/head.py`

```python
@runtime_checkable
class ClassifierHeadPort(Protocol):
    @property
    def identifier(self) -> str: ...
    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[Sequence[int]],
    ) -> None: ...
    def predict_proba(
        self,
        features: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]: ...
```

Multi-label: returns per-class probabilities, not softmax over classes.

### 4.5 `CalibratorPort`

`harness/ports/calibrator.py`

```python
@runtime_checkable
class CalibratorPort(Protocol):
    @property
    def identifier(self) -> str: ...
    def fit(
        self,
        oof_probabilities: Probabilities,
        labels: Sequence[Sequence[int]],
    ) -> None: ...
    def transform(self, probabilities: Probabilities) -> Probabilities: ...
```

Per-class calibration. `fit` is called once on validation OOF; `transform` is applied to val and test.

### 4.6 `ThresholdPort`

`harness/ports/threshold.py`

```python
@runtime_checkable
class ThresholdPort(Protocol):
    @property
    def identifier(self) -> str: ...
    def fit(
        self,
        calibrated_oof: Probabilities,
        labels: Sequence[Sequence[int]],
        *,
        config: ThresholdConfig,
    ) -> ThresholdSet: ...
    def apply(
        self,
        probabilities: Probabilities,
        thresholds: ThresholdSet,
    ) -> Predictions: ...
```

Contract: `fit` performs PR-sweep per class, applies shrinkage toward a prior (e.g. 0.5) by `config.shrinkage`, then clamps to `[clamp_lo, clamp_hi]`.

### 4.7 `MetricsPort`

`harness/ports/metrics.py`

```python
@runtime_checkable
class MetricsPort(Protocol):
    def evaluate(
        self,
        probabilities: Probabilities,
        labels: Sequence[Sequence[int]],
        thresholds: ThresholdSet,
        *,
        bootstrap: BootstrapConfig,
    ) -> MetricReport: ...
```

Returns macro and per-class F1/AUROC/AUPRC, all with bootstrap CIs at `bootstrap.confidence`.

### 4.8 `ArtifactStorePort`

`harness/ports/store.py`

```python
@runtime_checkable
class ArtifactStorePort(Protocol):
    def write_model_card(self, card: ModelCard, *, root: str) -> str: ...
    def write_data_card(self, dataset: Dataset, split: Split, *, root: str) -> str: ...
    def write_predictions(self, predictions: Predictions, *, root: str, name: str) -> str: ...
    def write_probabilities(self, probabilities: Probabilities, *, root: str, name: str) -> str: ...
    def write_thresholds(self, thresholds: ThresholdSet, *, root: str) -> str: ...
    def write_weights(self, weights_blob: bytes, *, root: str, name: str) -> str: ...
    def read_blob(self, uri: str) -> bytes: ...
```

Each `write_*` returns the URI under which the artifact was stored. URIs are opaque strings; the in-memory fake uses `mem://...`, the fs adapter uses `file://...`.

### 4.9 `RandomnessPort`

`harness/ports/randomness.py`

```python
@runtime_checkable
class RandomnessPort(Protocol):
    def seed_all(self, seed: int) -> None: ...
    def integers(self, low: int, high: int, size: int, *, seed: int) -> tuple[int, ...]: ...
    def child_seed(self, parent_seed: int, label: str) -> int: ...
```

`seed_all` seeds every random source the adapter is responsible for (Python `random`, numpy, optionally torch). `child_seed` derives deterministic sub-seeds (used to seed bootstrap, splitter, head independently from the master seed).

---

## 5. Adapter Matrix

Legend: **R** required for v1 publication MVP, **N** nice-to-have, **L** later (post-v1), **n/a** not applicable.

| Port               | Fake | sklearn | torch | fs   |
| ------------------ | ---- | ------- | ----- | ---- |
| DatasetPort        | R    | n/a     | L     | N    |
| SplitterPort       | R    | R       | n/a   | n/a  |
| BackbonePort       | R    | n/a     | L     | n/a  |
| ClassifierHeadPort | R    | R       | L     | n/a  |
| CalibratorPort     | R    | R       | n/a   | n/a  |
| ThresholdPort      | R    | R       | n/a   | n/a  |
| MetricsPort        | R    | R       | n/a   | n/a  |
| ArtifactStorePort  | R    | n/a     | n/a   | R    |
| RandomnessPort     | R    | R       | L     | n/a  |

### 5.1 v1 publication MVP set

The v1 runner can be assembled from:

- `InMemoryFakeDataset` *or* a thin sklearn-side adapter that reads a CSV manifest + pre-extracted features (no torch). For the publication the dataset adapter that consumes a CSV manifest plus pre-extracted feature `.npy` is acceptable and lives in `adapters/sklearn/dataset.py` (added in a follow-up PR; not required for the contract surface).
- `IterativeStratificationSplitter` (sklearn).
- `BackbonePort`: v1 uses a *precomputed-features* path. The fake backbone stands in for tests; the publication run pre-computes features offline and the dataset adapter ships them. A real torch backbone is L.
- `SklearnGBTHead` (gradient boosted trees per label, calibrated). `SklearnLogisticHead` is also R as a baseline.
- `IsotonicCalibrator` (R), `SigmoidCalibrator` (N).
- `PrSweepThreshold` (R).
- `SklearnMetrics` (R).
- `LocalFsArtifactStore` (R).
- `NumpySeededRandomness` (R).

### 5.2 Why the torch column is mostly L

Torch is gated behind the `harness-torch` extras group in `pyproject.toml` (added in a separate PR by a later agent — *not* by this spec). v1 tests must not import torch. v1 publication run uses precomputed features; the backbone adapter that drives torch lives in `adapters/torch/` and is exercised by a separate, opt-in test marker (`@pytest.mark.torch`) added later.

---

## 6. Composition Root

`harness/composition/runner.py` exposes `run_experiment`. This is the *only* function downstream consumers (notebooks, CLI scripts) call.

### 6.1 Signature

```python
def run_experiment(
    config: ExperimentConfig,
    *,
    dataset: DatasetPort,
    splitter: SplitterPort,
    backbone: BackbonePort,
    head: ClassifierHeadPort,
    calibrator: CalibratorPort,
    thresholds: ThresholdPort,
    metrics: MetricsPort,
    store: ArtifactStorePort,
    randomness: RandomnessPort,
) -> ExperimentResult: ...
```

All ports are keyword-only. No defaults — composition decides what to inject.

### 6.2 Algorithm (deterministic)

1. `randomness.seed_all(config.seed)`.
2. `ds = dataset.load()`. Validate `ds.label_names == config.label_names`.
3. `split = splitter.split(ds, val_fraction=..., test_fraction=..., seed=randomness.child_seed(config.seed, "split"))`.
4. Extract features for train, val, test via `backbone.extract(...)` over batches of `dataset.get_image_bytes(...)`.
5. `head.fit(train_features, train_labels)`.
6. `val_raw = head.predict_proba(val_features)` -> wrap into `Probabilities`.
7. `calibrator.fit(val_raw, val_labels)`.
8. `val_calibrated = calibrator.transform(val_raw)`.
9. `threshold_set = thresholds.fit(val_calibrated, val_labels, config=config.threshold)`.
10. `test_raw = head.predict_proba(test_features)`.
11. `test_calibrated = calibrator.transform(test_raw)`.
12. `test_preds = thresholds.apply(test_calibrated, threshold_set)`.
13. `report = metrics.evaluate(test_calibrated, test_labels, threshold_set, bootstrap=config.bootstrap)`.
14. Build `ModelCard` (deterministic config_hash from `config`).
15. Persist via `store.*` and collect `artifact_uris`.
16. Return `ExperimentResult(...)`.

### 6.3 Determinism rules

- The runner derives all sub-seeds via `randomness.child_seed(config.seed, label)` with stable labels: `"split"`, `"head"`, `"calibrator"`, `"threshold"`, `"bootstrap"`.
- The runner never calls `random.*`, `time.time()`, `uuid4()`, or `os.environ` directly. Anything time-like comes from a clock injected into `ModelCard` construction (v1: a single `created_at` injected by the composition root via factory).
- Adapters must be pure functions of `(input, seed)` where seed is supplied. Hidden RNG state is forbidden.

### 6.4 Factories

`harness/composition/factories.py`:

```python
def build_v1_runner_with_fakes(*, seed: int) -> Callable[[ExperimentConfig], ExperimentResult]: ...
def build_v1_runner_sklearn(*, seed: int, artifact_root: str) -> Callable[[ExperimentConfig], ExperimentResult]: ...
```

Factories construct the adapter set, partial-apply `run_experiment`, and return a callable. Factories are the *only* place adapters are instantiated outside tests.

---

## 7. Test Pyramid

Test root: `tests/harness/`. Three tiers, three responsibilities.

### 7.1 Contract tests

One test class per port. Each class is generic over an adapter, parametrized via a fixture.

Pattern:

```python
# tests/harness/contract/test_threshold_port_contract.py

class ThresholdPortContractTests:
    """Abstract contract; subclasses provide `adapter` fixture."""

    @pytest.fixture
    def adapter(self) -> ThresholdPort:
        raise NotImplementedError

    def test_fit_returns_one_threshold_per_label(self, adapter: ThresholdPort) -> None: ...
    def test_thresholds_are_clamped(self, adapter: ThresholdPort) -> None: ...
    def test_apply_produces_zero_or_one(self, adapter: ThresholdPort) -> None: ...
    def test_apply_is_deterministic(self, adapter: ThresholdPort) -> None: ...
    def test_fit_is_seed_stable_when_applicable(self, adapter: ThresholdPort) -> None: ...

class TestFixedFakeThresholdContract(ThresholdPortContractTests):
    @pytest.fixture
    def adapter(self) -> ThresholdPort:
        return FixedFakeThreshold(value=0.5)

class TestPrSweepThresholdContract(ThresholdPortContractTests):
    @pytest.fixture
    def adapter(self) -> ThresholdPort:
        return PrSweepThreshold()
```

Required contract test classes:

- `DatasetPortContractTests`
- `SplitterPortContractTests`
- `BackbonePortContractTests`
- `ClassifierHeadPortContractTests`
- `CalibratorPortContractTests`
- `ThresholdPortContractTests`
- `MetricsPortContractTests`
- `ArtifactStorePortContractTests`
- `RandomnessPortContractTests`

Every adapter must be wired into its port's contract suite. Contract tests assert *behavior* (shape, invariants, determinism), not specific numeric values where adapters legitimately differ.

Universal contract assertions (apply to most ports):

- Determinism given the same seed and inputs.
- Shape and dtype of outputs.
- Invariants from section 3 hold on outputs.
- Errors raised through `HarnessError` hierarchy.

### 7.2 Unit tests

Algorithmic correctness, per adapter, on synthetic data with closed-form expected answers.

Required unit tests:

- `tests/harness/unit/adapters/sklearn/test_pr_sweep_threshold.py`
  - Synthetic 1-class case where the F1-maximizing threshold is known by construction; assert `PrSweepThreshold.fit` finds it within a tolerance equal to the sweep step.
  - Shrinkage moves the threshold toward 0.5 by exactly `shrinkage * (t* - 0.5)`.
  - Clamps respected at the boundaries.
- `tests/harness/unit/adapters/sklearn/test_isotonic_calibrator.py`
  - Monotonicity: if raw `p1 <= p2`, calibrated `q1 <= q2` per class.
  - On a perfectly miscalibrated synthetic set (e.g. `p = sigmoid(2 * logit(true_p))`), reduces ECE.
- `tests/harness/unit/adapters/sklearn/test_sklearn_metrics.py`
  - Hand-built confusion: macro-F1 matches a manual computation.
  - Bootstrap CI width shrinks as `n_resamples` grows (sanity, not exact).
  - CI coverage on synthetic Bernoulli labels lands within tolerance.
- `tests/harness/unit/adapters/sklearn/test_iterative_splitter.py`
  - No patient leakage across train/val/test.
  - All labels appear in train (when n is large enough) — soft assertion via tolerance.
  - Same seed -> same split.
- `tests/harness/unit/adapters/fakes/test_in_memory_dataset.py`
  - Round-trips samples and label names verbatim.
  - `get_image_bytes` returns deterministic synthetic bytes for a given `image_ref`.
- `tests/harness/unit/adapters/fakes/test_seeded_randomness.py`
  - `child_seed` is deterministic and collision-free for distinct labels.
- `tests/harness/unit/adapters/fs/test_local_fs_store.py`
  - Round-trips ModelCard / Predictions / ThresholdSet through write + read_blob.
  - URIs returned are well-formed `file://` URIs.

Domain unit tests:

- `tests/harness/unit/domain/test_types.py`
  - `Dataset` rejects samples whose `labels` length disagrees with `label_names`.
  - `ThresholdSet` rejects out-of-range thresholds.
  - `Probabilities` rejects values outside `[0, 1]`.

### 7.3 Integration tests

Full runner end-to-end with the fake adapter set.

- `tests/harness/integration/test_runner_golden_path.py`
  - Build runner via `build_v1_runner_with_fakes(seed=...)`.
  - Run on a fixed fake dataset (e.g. 64 samples, 6 labels, 16 patients).
  - Assert `ExperimentResult` shape: every field populated, `report.per_class` length matches `label_names`, `artifact_uris` includes keys `model_card`, `data_card`, `thresholds`, `val_probabilities`, `test_probabilities`, `test_predictions`.
- `tests/harness/integration/test_runner_reproducibility.py`
  - Run the runner twice with the same `config.seed` and the same fake adapters.
  - Assert byte-equal `ThresholdSet`, byte-equal `Predictions`, and equal `MetricReport` (point + interval bounds).

Integration tests must not write to the real filesystem; they use `InMemoryFakeStore`. A separate fs-store integration test (under unit/fs) covers real disk I/O at small scale.

### 7.4 Test isolation

- All tests use `pytest`'s `tmp_path` for any disk activity.
- Tests must not rely on network. CI is offline.
- No test takes longer than 1 second except the integration golden path (budget: 3 seconds).
- Total `tests/harness/` wall time budget: 10 seconds locally, no torch.

### 7.5 Markers

- `@pytest.mark.torch` for any test that needs torch (none in v1).
- `@pytest.mark.slow` for tests over the 1-second budget (only the integration golden path qualifies).

---

## 8. TDD Rules of Engagement

These are non-negotiable rules every downstream agent must follow when contributing to `harness/`.

### 8.1 Red, green, refactor — actually red first

1. Write the failing test first.
2. Run the test, observe it fail with the *expected* failure mode (not an import error). Capture the failure in the PR description.
3. Implement the minimum code to make it pass.
4. Refactor with the test as a safety net.

A PR whose first commit adds passing tests alongside implementation will be rejected.

### 8.2 Tests assert behavior, not implementation

- No `assert isinstance(x, ExpectedClass)` as the only assertion. That tests the type system, not behavior.
- No `mock.assert_called_with(...)` as the primary assertion when a real fake exists. Use the fake and assert observable effects.
- Prefer assertions over equality of full structures when the field set is large; assert the fields you care about and their invariants.

### 8.3 Coverage rules

- Every public function in `harness/` gets at least one unit test.
- Every port gets a contract test class. Every adapter implementing a port gets a concrete subclass plugged into that contract suite.
- Every runner factory gets one integration test.

### 8.4 Type hygiene

- No `Any`. If a third-party library leaks `Any`, narrow it at the adapter boundary with `cast` and a justified comment.
- No bare `# type: ignore`. Always `# type: ignore[error-code]  # reason: ...`.
- mypy `--strict` must pass on `harness/` and `tests/harness/`.

### 8.5 Import hygiene

- `from torch import ...` is illegal anywhere except `harness/adapters/torch/`. A repo-level lint check (added by a later agent) enforces this; until then, code review enforces it.
- `from harness.adapters.* import ...` is illegal in `harness/domain/`, `harness/ports/`, and other adapter packages.
- `harness/composition/` is the only place that imports from multiple adapter packages.

### 8.6 Determinism

- No wall-clock dependence in adapters. Inject a clock at the composition root.
- No `random` / `np.random` / `torch.*` global state. Always use the `RandomnessPort` or a seed parameter.
- No `os.environ` reads in adapters. Config flows in via `ExperimentConfig` or factory args.

### 8.7 Errors

- Adapters raise `AdapterError` (or subclass) on adapter-specific failures.
- Domain validation raises `DomainValidationError` from `__post_init__`.
- Contract tests assert that adapters never raise `Exception` directly.

### 8.8 Public surface

- `harness/__init__.py` exports only: domain types, ports, `run_experiment`, and the v1 factories.
- Adapters are not part of the public surface. Consumers import them only via factories.

### 8.9 Code review checklist (must pass before merge)

- [ ] New tests landed first commit, demonstrably failed before implementation.
- [ ] Contract tests parametrized over every adapter that implements the port.
- [ ] mypy `--strict` clean.
- [ ] ruff clean.
- [ ] No torch import outside `adapters/torch/`.
- [ ] No `Any`, no bare `# type: ignore`.
- [ ] Layering rules respected (no upward imports).
- [ ] Determinism: same seed -> same outputs verified by an integration or contract test.

---

## 9. Non-Goals for v1

The following are explicitly *out of scope* for v1 and must not be smuggled in:

- **Real torch training.** No backbone fine-tuning. No optimizer loop. The torch backbone, when added, only does inference on a frozen pretrained network and lives behind the `harness-torch` extras flag.
- **Real NIH download.** No HTTP, no kaggle CLI, no disk scraping in the harness. The publication run uses a CSV manifest of pre-extracted features, ingested via a future `adapters/sklearn/dataset.py`. v1 tests use `InMemoryFakeDataset` exclusively.
- **GPU code paths.** Everything in v1 runs on CPU.
- **Multi-node orchestration.** Single-process, single-machine.
- **Online learning / streaming.** Batch only.
- **A web UI.** Reports are written as artifacts via `ArtifactStorePort`. Rendering them is a separate concern.
- **Replacing `analyzer/`.** `harness/` is parallel. Cross-imports between the two modules are forbidden.
- **Hyperparameter search.** v1 fixes hyperparameters in `ExperimentConfig`. A future `SearchPort` may be added.

---

## 10. Glossary

- **OOF**: out-of-fold. Validation predictions used to fit calibration and tune thresholds without touching test data.
- **PR-sweep**: enumerating thresholds along the precision-recall curve to find the F1-maximizing operating point per class.
- **Shrinkage**: regularizing the chosen threshold toward a prior (0.5) to reduce variance from small validation sets. Implemented as `t' = t + s * (0.5 - t)` for `s in [0, 1]`.
- **Clamp**: bounding the final threshold to `[clamp_lo, clamp_hi]` to avoid degenerate operating points (e.g. always-positive or always-negative).
- **Patient-level split**: each patient's samples appear in exactly one of train/val/test; prevents leakage from same-patient correlated images.
- **Multi-label stratification**: balances label prevalence across splits subject to the patient-level constraint. Implemented via iterative stratification (Sechidis et al., 2011).
- **Bootstrap CI**: nonparametric confidence interval from `n_resamples` resamples of the test set with replacement.
- **Config hash**: SHA-256 over a canonical JSON serialization of `ExperimentConfig`, used as a primary key for artifacts.

---

## 11. Open questions deferred to follow-up specs

These are *deliberately* unresolved in v1 and require their own spec PRs:

1. Real CSV manifest dataset adapter (sklearn-side) — schema, label vocabulary mapping, missing-label policy.
2. Torch backbone adapter — which pretrained network, image preprocessing pipeline, batch-size policy.
3. Hyperparameter search port and adapter.
4. Result registry (a queryable index of `ExperimentResult` records, separate from the artifact store).
5. CLI entry point (likely `python -m harness run --config path.toml`).

Each of the above gets its own ARCHITECTURE-style spec under `harness/docs/` before any code lands.

---

## 12. Source-of-truth contract

This document is the single source of truth for `harness/`. If code disagrees with this doc, the code is wrong unless a spec PR has updated this doc first. Downstream agents must:

1. Read this document end-to-end before contributing.
2. Cite the relevant section in PR descriptions.
3. Open a spec PR (modifying this file) before making any architectural change.

End of spec.

---

## 13. v1 Implemented Surface

This section documents *adopted* port and domain signatures as they actually
landed in `harness/` for v1. Earlier sections (§3 / §4 / §6) describe the
spec as drafted; the implementation deviated in narrowly-scoped, deliberate
ways to keep the code numpy-native, framework-agnostic, and within strict
mypy. **When the rest of this document disagrees with section 13, section 13
wins for v1.** Spec PRs that re-align the upper sections may land later.

### 13.1 Adopted-vs-spec table

| Original spec (§N) | v1 actual | Rationale |
| ------------------ | --------- | --------- |
| §3 preamble: "numpy not allowed in `domain/`" | `harness/domain/types.py` imports `numpy` and uses `NDArray[np.float32]` / `NDArray[np.int8]` for `Probabilities.values` and `Predictions.values`. | Documented exception captured in the module docstring. numpy is already a hard project dependency, every adapter speaks it at its boundary, and it lets `__post_init__` validate shape / range invariants vectorially. No other third-party imports are permitted in `domain/`. |
| §3.4 `Probabilities.values: tuple[tuple[float, ...], ...]` | `Probabilities.values: NDArray[np.float32]` | Same rationale as above; nested tuples would force every adapter to round-trip through Python objects and lose dtype/shape guarantees. The `__post_init__` checks ndim, shape match against `(sample_ids, label_names)`, and `[0, 1]` value range. |
| §3.5 `Predictions.values: tuple[tuple[int, ...], ...]` | `Predictions.values: NDArray[np.int8]` | Mirror of `Probabilities`; `__post_init__` enforces 0/1 only. |
| §4.3 `BackbonePort.feature_dim` | `BackbonePort.embedding_dim` | Renamed for consistency with the broader ML embedding vocabulary (`embedding_dim` is what callers expect). |
| §4.3 `BackbonePort.extract(image_bytes_batch: Sequence[bytes]) -> tuple[tuple[float, ...], ...]` | `BackbonePort.extract(images: NDArray[np.float32]) -> NDArray[np.float32]` (shape `(N, H, W, C)` -> `(N, embedding_dim)`) | Adapters convert raw `bytes` to a tensor *before* calling `extract`; keeps the port numpy-native and torch/sklearn agnostic. Conversion lives in the composition root (`_bytes_to_image_tensor` in `harness/composition/runner.py`). The port no longer needs a `Sequence[bytes]` type, which would have forced numpy/torch backends to round-trip through Python lists. |
| §4.3 `BackbonePort.identifier` | Not present on the port (some adapters expose it; the runner's `_describe_port` helper falls back gracefully). | Kept identifier as an *adapter* concern, not a port-level requirement, since the runner already discovers it via `getattr(port, "identifier", None)`. |
| §4.4 `ClassifierHeadPort.fit/predict_proba` typed as `Sequence[Sequence[...]]` and returns `tuple[tuple[float, ...], ...]` | All inputs / outputs are numpy arrays: `NDArray[np.float32]` features, `NDArray[np.int8]` labels, `NDArray[np.float32]` probabilities. | sklearn / torch heads natively produce numpy. Round-tripping through nested tuples adds zero safety (mypy can't check inner shape) and measurable cost. |
| §4.4 `ClassifierHeadPort.identifier` | Not on the port; same fallback pattern as backbone. | Same rationale as §4.3 identifier. |
| §4.5 `CalibratorPort.fit(oof_probabilities: Probabilities, labels: Sequence[Sequence[int]])` and `transform(probabilities: Probabilities) -> Probabilities` | `CalibratorPort.fit(probs: NDArray[np.float32], labels: NDArray[np.int8])` and `transform(probs: NDArray[np.float32]) -> NDArray[np.float32]`; the runner wraps results into the domain `Probabilities` type at the composition boundary. | Domain `Probabilities` already wraps an `NDArray` internally; ports work directly with the underlying tensor. The composition root re-wraps before persistence. Calibrators also expose `is_fitted: bool` so contract tests can assert the "transform-before-fit" failure mode. |
| §4.5 `CalibratorPort.identifier` | Not on the port. | Same rationale as backbone/head. |
| §4.6 `ThresholdPort` | Matches spec except for the `identifier` property, which **is** present on the threshold port (used in `ThresholdSet.method`). | Threshold method is part of the persisted artifact, so identifier must be a first-class port concern. |
| §4.7 `MetricsPort` | Matches spec. | No drift. |
| §4.8 `ArtifactStorePort` -- 7 methods (`write_model_card`, `write_data_card`, `write_predictions`, `write_probabilities`, `write_thresholds`, `write_weights`, `read_blob`) | v1 ships 4 methods: `write_model_card`, `write_thresholds`, `write_metric_report`, `write_predictions`. The remaining methods are **DEFERRED to v1.1**: `write_data_card`, `write_probabilities`, `write_weights`, `read_blob`. | The runner currently produces only the four artifacts the v1 publication needs (model card, thresholds, test predictions, metric report). Data cards, raw probabilities, frozen weights, and `read_blob` are real future requirements but adding them now would require corresponding fake/store implementations and contract tests for surfaces that no caller exercises. Tracked for v1.1; the slice's docstring on `ArtifactStorePort` documents this. |
| §4.9 `RandomnessPort` | Matches spec. | No drift. |
| §6.1 Runner signature | Matches spec, plus an optional `clock: datetime \| None = None` keyword for deterministic `created_at` injection. | Required to make `test_runner_reproducibility.py` byte-stable; the spec hand-waved "a clock injected into ModelCard construction" -- this is the concrete hook. Default behaviour falls back to `datetime.now(tz=UTC)` when no clock is supplied. |
| §3.3 Splitter contract: "union is a *subset* of `range(n_samples)`" | v1 contract test asserts **equality**: `union(train, val, test) == set(range(n_samples))`. | A splitter that drops samples silently is a worse failure mode than one that errors; v1 commits to "no sample left behind." Adapters that need to reject samples must fail loudly during `split`. |
| §6.2 step 7: "calibrator.fit(val_raw, val_labels)" plus implicit OOF semantics | v1 fits the calibrator and the threshold tuner on the **same val fold** (both held out from training). Strict OOF (k-fold CV across the train set) is **deferred to v1.1**. | A k-fold OOF loop multiplies head-fit cost by `k` and would balloon the integration suite well beyond its 3-second budget. v1 commits to "calibrator + threshold are co-fit on a single val fold," which is the standard practice when validation set is large enough; the bias from re-using val for two stages is documented in the model card's `notes` field. |
| §3.9 `BootstrapConfig.confidence` | `0.0 < confidence < 1.0` (open interval), with a one-line comment in `__post_init__`. | Open-interval matches the bootstrap library's domain (degenerate at the endpoints) and prevents subtly broken downstream CI math. |

### 13.2 Public surface (`harness/__init__.py`)

v1 re-exports exactly:

* `run_experiment` (composition root)
* `build_v1_runner_with_fakes`, `build_v1_runner_sklearn` (factories)
* All domain types from `harness.domain` (the full `__all__` of that module)

Ports and adapter classes are **not** part of the public surface. They are
internal extension points; consumers wire them via the factories.

### 13.3 Factories return `RunnerBundle`, not `dict[str, object]`

`build_v1_runner_with_fakes` and `build_v1_runner_sklearn` return a frozen
`RunnerBundle` dataclass with explicitly-typed port fields rather than a
`dict[str, object]`. This removes the `cast()` calls that integration tests
previously needed and tightens mypy coverage at the factory/runner boundary.
The bundle's fields are: `config`, `dataset`, `splitter`, `backbone`, `head`,
`calibrator`, `thresholds`, `metrics`, `store`, `randomness`.

### 13.4 Error type for adapter init failures

`FixedFakeThreshold.__init__` (and any other adapter that validates its
constructor arguments) raises `AdapterError` (or `ConfigError` if the
argument represents user-supplied configuration) -- **not**
`ContractViolation`. `ContractViolation` is reserved for runtime port-contract
violations during operation (shape mismatch, value out of range, etc.), not
for invalid init arguments. Contract tests assert this distinction.

### 13.5 Torch backbone adapter (v1)

The `harness/adapters/torch/` package ships two `BackbonePort` implementations
in v1. They are framework-gated behind the `[experiment]` extras group and
excluded from the default test suite via the `torch` pytest marker.

**Classes shipped.**

| Class | Underlying network | Embedding dim |
| ----- | ------------------ | ------------- |
| `TorchVisionResNet50Backbone` | `torchvision.models.resnet50` (`IMAGENET1K_V2` default, falls back to `V1`) | `(N, 2048)` |
| `TorchVisionDenseNet121Backbone` | `torchvision.models.densenet121` (`IMAGENET1K_V1`) | `(N, 1024)` |

**Preprocessing pipeline.** Input is `NDArray[np.float32]` with shape
`(N, H, W, C)` and values in `[0, 1]`. The pipeline:

1. Validates ndim == 4 and `C in {1, 3}`; other channel counts raise
   `AdapterError`.
2. Permutes NHWC -> NCHW.
3. If `C == 1`, replicates along the channel axis to produce 3 channels
   (chest X-rays are grayscale; the ImageNet-trained backbone wants RGB).
4. Bilinear-resizes to `224 x 224` (`align_corners=False`).
5. Normalizes with the standard ImageNet mean / std
   (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

A unit test (`test_one_channel_input_matches_three_channel_replication`)
asserts that `extract(gray)` matches `extract(np.repeat(gray, 3, axis=-1))`
within `rtol=1e-5, atol=1e-6`, guarding against silent drift in the
replication path.

**Device fallback.** Auto-selection (`device=None`) picks
`mps` -> `cuda` -> `cpu`. MPS is claimed only when both
`torch.backends.mps.is_available()` *and* `torch.backends.mps.is_built()`
return `True`. A caller-supplied override (`device="cuda" | "mps" | "cpu"`)
is validated against the runtime; an override that is not present raises
`AdapterError` rather than silently falling back. Unknown override strings
also raise.

**Eval-only.** `model.eval()` is called once at construction; the
classification head (`model.fc` for ResNet, `model.classifier` for DenseNet)
is replaced with `nn.Identity` so `forward` returns penultimate features
directly. `extract` runs inside `torch.no_grad()`. There is no training
surface exposed.

**Output.** `NDArray[np.float32]` of shape `(N, embedding_dim)`. Features
are detached, moved to the host, and dtype-cast at the boundary.

**Determinism.** `torch.manual_seed(seed)` is called *inside* the
constructor, never globally. Two adapters built with the same seed and the
same `weights` value produce byte-identical features for the same input.
**v1 deviation:** `torch.manual_seed` mutates global torch RNG state. This
is acceptable for v1 because (1) the seed is consumed only at construction
time for random-weight init paths used by tests; (2) inference runs under
`torch.no_grad()` and does not consume RNG. v1.1 may migrate to a
`torch.Generator`-scoped pattern.

When the chosen device is `cuda`, the constructor also sets
`torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False`. MPS does not currently provide an
equivalent determinism flag; runs on MPS are not byte-reproducible across
devices.

The `weights` parameter uses a private `_DefaultWeights` sentinel class
(not a bare `object()`) to distinguish "caller did not pass `weights=`"
from "caller passed `weights=None` for random init." This keeps the
constructor signature mypy-narrow.

**Deferred (v1.1+).**

* `RadImageNetResNet50Backbone` (a CXR-pretrained ResNet50 variant) — separate PR.
* Composition factory wiring (Step 3 of `PAPER_CHECKLIST.md`); the torch adapters are not yet returned by `build_v1_runner_*`.
* Real-weight smoke tests (the unit suite uses `weights=None`).

### 13.6 Publication factory wiring (Step 3)

```harness/composition/factories.py``` ships a third factory in v1:

```python
def build_publication_runner_v1(
    seed: int,
    *,
    nih_csv_path: Path,
    nih_images_dir: Path,
    artifact_root: Path,
) -> RunnerBundle: ...
```

It wires the real on-disk pipeline: `NIHDataset` ->
`IterativeStratifiedPatientSplitter` -> `TorchVisionResNet50Backbone`
(default ImageNet weights, auto device) -> `SklearnGradientBoostingHead` ->
`PerClassIsotonicCalibrator` -> `PrSweepShrinkageThreshold` ->
`BootstrapMetrics` -> `FilesystemArtifactStore`. Per the §6.4 factory
contract, sub-seeds are derived through `SeededRandomness.child_seed`
with stable labels (`"backbone"`, `"head"`, `"bootstrap"`).

Two divergences from the §6.4 fakes/sklearn factories are worth flagging.

**(1) Decoding wrappers between the dataset and the torch backbone.** The
runner's `_bytes_to_image_tensor` materialises every byte blob into a
`(N, 4, 4, 2)` tensor (`BYTES_IMAGE_SHAPE`) before calling the backbone.
That shape suits the fake adapter set but rejects ResNet50, which requires
`C in {1, 3}`. Because `runner.py` is the stable single-source-of-truth
for the experiment loop, the publication factory bridges the mismatch with
two private wrappers in `harness/composition/_publication_pipeline.py`:

* `_DecodingDataset` decodes each PNG via `NIHImageLoader.decode` to a
  `(H, W, 1) float32 [0, 1]` tensor at `get_image_bytes` time, stashes
  the array in a per-run `DecodedImageCache` keyed by `sha256(ref)[:32]`,
  and returns those 32 bytes as the runner-visible blob. The byte-blob
  round-trips exactly through `uint8 -> /255.0 -> *255.0 -> round` in
  float32 (verified at composition time).
* `_DecodingBackbone` recovers the cache key from each `(4, 4, 2)` row,
  stacks the cached decoded images into `(N, H, W, 1)`, and forwards to
  the wrapped `TorchVisionResNet50Backbone`.

The wrappers implement the existing port protocols verbatim and are
composition-internal; no new port surface is introduced. Lifetime of the
side-channel cache is bounded by a single `run_experiment` call. v1.1 may
refactor the runner to push bytes-to-tensor responsibility into the dataset
adapter and remove the indirection.

**(2) Adapter substitutions.** Three names from
`PAPER_CHECKLIST.md` Step 3 do not exist verbatim in the codebase. The
factory uses the implemented adapters and documents the substitution:

| Brief calls for | Implemented as | Notes |
| --------------- | -------------- | ----- |
| `SklearnGBTHead` | `SklearnGradientBoostingHead` | Shipped this PR; `HistGradientBoostingClassifier`-backed, one per-label fit, deterministically seeded via `RandomnessPort.child_seed(seed, "head")`. |
| `IsotonicCalibrator` | `PerClassIsotonicCalibrator` | Same algorithm, project naming. |
| `PrSweepThreshold` | `PrSweepShrinkageThreshold` | Same algorithm; the implemented name advertises the shrinkage component. |
| `SklearnMetrics` | `BootstrapMetrics` | Same scope (macro/per-class F1/AUROC/AUPRC + bootstrap CIs). |
| `LocalFsArtifactStore` | `FilesystemArtifactStore` | Same role. |
| `NumpySeededRandomness` | `SeededRandomness` (in `adapters/fakes/`) | Numpy-backed, seed-pure. The package placement reflects the v1 module layout; the implementation is production-grade. |

`build_publication_runner_v1` therefore satisfies the spirit of §6.4 --
factories are the only place adapters are instantiated outside tests, and
the bundle is byte-identical in shape to `build_v1_runner_sklearn`.

### 13.7 Step 3.5 — Feature cache + ablation runner

Step 3.5 in `PAPER_CHECKLIST.md` adds two coupled pieces on top of the
publication factory: a content-addressable feature cache (so the expensive
ResNet50 forward pass runs at most once per unique input image) and a
seed-grid ablation CLI (so multi-variant comparisons reuse that cache).

**`CachedBackbone` adapter (`harness/adapters/fs/cached_backbone.py`).**

* Wraps any `BackbonePort`. Inner adapter must expose a non-empty `identifier`
  string; without one the cache cannot scope its layout safely (a backbone
  swap would silently return stale features). Construction raises
  `AdapterError` if the identifier is missing.
* Cache key is `sha256(image.tobytes())` over each row of the input
  `(N, H, W, C) float32` tensor — content-addressable, so the same image
  bytes produce the same cache file regardless of the source CSV /
  `images_dir`.
* On-disk layout: `{cache_dir}/{backbone_id}/{sha[:2]}/{sha}.npy`. Two-char
  prefix sharding keeps any single directory under ~256 children even at the
  full 112k NIH corpus.
* Atomic writes via `*.npy.tmp` + `Path.replace`; a crash mid-write leaves no
  partial `.npy` for a future run to mistakenly load.
* Per-row hashing with batched miss extraction: rows that hit the cache are
  read directly; misses are stacked into a single batch and forwarded to the
  inner backbone in one call (preserving input order in the output).
* The wrapper's own `identifier` is `cached:<inner_id>` so downstream
  consumers (model card, lineage tracking) see the cache layer in the
  recorded backbone identifier.

**Factory parameter: `feature_cache_dir`.**

`build_publication_runner_v1(..., feature_cache_dir: Path | None = None)`:

* `None` (default): no caching, every run re-extracts (preserves the prior
  Step 3 behaviour byte-for-byte).
* `Path`: the factory creates the directory if it does not exist (`mkdir
  parents=True, exist_ok=True`), validates that an existing path is a
  directory (raising `ConfigError` otherwise), and wraps the inner
  `TorchVisionResNet50Backbone` in `CachedBackbone` **before** the
  `_DecodingBackbone` wrapper. Wrap order matters: cache writes happen on
  the resized `(N, 224, 224, 1)` tensor that ResNet50 actually consumes,
  not on the runner's `(4, 4, 2)` byte-key tensor (which would be useless
  across runs).

**Ablation runner: `harness/scripts/run_ablation.py`.**

A CLI that runs the publication pipeline once per master seed against a
shared `feature_cache_dir`. v1 variant axis = master seed only. Args:
`--seeds` (required, comma-separated unique ints), `--nih-csv`,
`--nih-images`, `--n` (default 0 = full), `--feature-cache-dir`
(required — caching is the whole point), `--strict-missing-images`
(BooleanOptionalAction, default True), `--artifact-root`
(default `runs/ablation-<UTC-timestamp>/`).

Per-seed artifacts land under `<artifact-root>/seed-<n>/`. After every seed
runs, the script writes `<artifact-root>/comparison.csv` with columns:

```
seed, macro_f1, macro_f1_ci_low, macro_f1_ci_high, macro_auroc, macro_auprc
```

If a seed raises mid-run, the exception is logged to stderr (`seed=<n>
failed: <repr>`) and the loop continues with the remaining seeds. The
script exits 1 if any seed failed, 0 otherwise. `comparison.csv` includes
only the seeds that completed successfully.

**v1 deviations.**

* The cache is single-process; there is no inter-process locking. Concurrent
  ablation runs against the same `feature_cache_dir` may duplicate work
  (atomic rename prevents corruption) but must not be relied on for
  correctness across truly-concurrent writers.
* Cache keys are content-addressable. The same image bytes produce the same
  cache file across CSVs, image directories, and pilot truncations. This is
  intentional and deduplicates work across runs; callers who want
  per-experiment isolation must supply distinct `feature_cache_dir` paths.
* No cache eviction. Bounded by dataset size in practice (≤ 112k entries
  for full NIH); explicit cache hygiene is the operator's responsibility.
* `comparison.csv` is overwritten each run if `artifact-root` is reused;
  the script does not version the file.

**Deferred to v1.1+.**

* Variant axes beyond seed (head, calibrator, threshold). When they land,
  the runner grows additional `--variant-*` flags rather than a new entry
  point.
* Concurrent-safe cache writes (file locking or per-writer scratch
  directories).
* Cache eviction policy (LRU, size-bounded).
* Distributed feature extraction (sharded across machines or processes).
* Patient-block-aligned `--n` truncation (currently inherits the same
  mid-patient truncation semantics as `run_pilot.py`).

### 13.8 v1.1 fine-tuning surface (design only; not yet implemented)

End-to-end fine-tuning is the load-bearing v1.1 capability: the architecture
that lets the harness train the backbone + head together on the full NIH-14
corpus rather than running frozen-feature extraction. The full design --
port surface, adapter signature, composition wiring, test strategy,
determinism story, and out-of-scope items -- lives in
`harness/docs/FINE_TUNING_DESIGN.md`.

**Surface introduced (design PR; stub only).**

* `harness/ports/trainer.py` -- three new Protocols:

  * `TrainingDatasetPort` -- finite-length iterable of decoded
    `(image, labels)` rows, built by the composition root from a
    `DatasetPort` + split index list.
  * `TrainedClassifierPort` -- the eval-ready model returned by the
    trainer; exposes only `predict_proba(images) -> probabilities` so the
    runner can route its output into the existing
    `CalibratorPort` -> `ThresholdPort` -> `MetricsPort` chain.
  * `TrainerPort` -- end-to-end trainer; `fit(*, training_dataset,
    validation_dataset, config, seed) -> tuple[TrainedClassifierPort,
    TrainingResult]`.

* `tests/harness/contract/test_trainer_contract.py` -- abstract
  `TrainerPortContract` base class plus a synthetic two-class `_TinyDataset`.
  No concrete subclasses yet; the implementation PR adds
  `TestTorchFineTuneTrainerContract` under TDD red-green discipline.

**Surface deferred to the implementation PR.**

* `TrainingConfig` and `TrainingResult` domain types
  (`harness/domain/types.py`).
* `TorchFineTuneTrainer` adapter (`harness/adapters/torch/trainer.py`).
* `run_finetune_experiment` runner entry point and `FineTuneRunnerBundle`
  bundle dataclass.
* `build_finetune_runner_v1` factory.
* Unit tests, integration test on the 16-row NIH fixture, and a
  `@pytest.mark.smoke` test on the 4999-row slice asserting
  macro-AUROC > 0.65 (the frozen-feature floor).

**Why a new port and not an extension of `BackbonePort`.** Per
FINE_TUNING_DESIGN.md §1, fine-tuning is fundamentally a different
operation from frozen feature extraction (it owns the optimizer, loss,
LR schedule, DataLoader, augmentation, checkpointing). Extending the
eval-only `BackbonePort` would force every existing adapter --
`TorchVisionResNet50Backbone`, `TorchVisionDenseNet121Backbone`,
`TXRVDenseNet121NIHBackbone`, `IdentityFakeBackbone`, `CachedBackbone` --
to stub a `fit()` method none of them can meaningfully implement.
Subclassing (`TrainableBackbonePort(BackbonePort)`) still smuggles
head + loss + optimizer concerns into "the backbone." A separate
`TrainerPort` keeps the existing frozen-feature pipeline byte-identical
and adds a clean parallel path.

**Why a new runner function and not a flag on `run_experiment`.** Per
FINE_TUNING_DESIGN.md §4, `run_experiment` defines all ports as required
keyword arguments per §6.1; adding optional ports + `if/else` branches
violates that contract. Two named entry points (`run_experiment` for
frozen-feature, `run_finetune_experiment` for fine-tune) with
non-overlapping bundle shapes is the cleaner split.


### 13.9 v1.1 fine-tuning surface (implemented)

End-to-end fine-tuning landed in PR pate/fine-tune-impl-v1, building on
the design + Protocol stubs from §13.8. The surface introduced is a
strict superset of §13.8's "design only" listing.

**Surface introduced (implementation PR).**

* `harness/domain/types.py` — `TrainingConfig` and `TrainingResult`
  frozen dataclasses with `__post_init__` validation per
  FINE_TUNING_DESIGN.md §3.1 / §3.2; `ExperimentConfig.training:
  TrainingConfig | None` field (default `None`) per §4.4.
* `harness/adapters/torch/trainer.py` — `TorchFineTuneTrainer` adapter
  implementing `TrainerPort`. Supports `densenet121` and `resnet50`
  backbones with ImageNet-pretrained init; unknown `backbone_id`
  (including any TXRV variant) raises `ConfigError`. Loss is
  `nn.BCEWithLogitsLoss`; optimizer is `torch.optim.AdamW`. LR schedule
  is cosine with linear warmup, or constant. Augmentations are
  `torchvision.transforms.v2`-based (`hflip` ->
  `RandomHorizontalFlip(p=0.5)`, `rotate10` -> `RandomRotation(degrees=10)`).
  Device ladder mirrors the eval-only adapters (mps -> cuda -> cpu).
  Checkpoints are `torch.save(dict, path)` per §10 answer #2.
* `harness/composition/_finetune_pipeline.py` —
  `_InMemoryTrainingDataset` (in-memory `TrainingDatasetPort`; 20000-row
  cap enforced at construction per §10 answer #3), `FineTuneRunnerBundle`
  dataclass, and an `ImageDecoder` callable type alias.
* `harness/composition/runner.py` — new `run_finetune_experiment`
  function. The existing `run_experiment` now raises `ConfigError` when
  `config.training is not None` so the two paths are mutually exclusive
  per §4.4.
* `harness/composition/factories.py` — new `build_finetune_runner_v1`
  factory returning a `FineTuneRunnerBundle`. Default training config
  uses CheXNet augmentations `("hflip", "rotate10")` per §10 answer #5.
* `tests/harness/contract/test_trainer_contract.py` — appended
  `TestTorchFineTuneTrainerContract` concrete subclass (CPU device, 4
  epochs, 32x32 input). Tagged `@pytest.mark.torch`.
* `tests/harness/unit/test_training_config.py`,
  `test_in_memory_training_dataset.py`, `test_torch_finetune_trainer.py`
  — RED-first validator + adapter unit suites.
* `tests/harness/integration/test_finetune_runner.py` — 16-row fixture
  end-to-end wiring test. `@pytest.mark.torch @pytest.mark.slow`.
* `tests/harness/integration/test_finetune_runner_smoke.py` — the
  load-bearing smoke gate per §10 answer #1: 1 epoch on the 4999-row
  NIH slice, asserts `report.macro_auroc.point > 0.65`.
  `@pytest.mark.smoke @pytest.mark.slow @pytest.mark.torch`. Observed
  smoke result on the implementing run: macro-AUROC = 0.7011.

**Approved §10 sign-offs (deviations from the design doc that landed
verbatim in the implementation PR).**

* §10 answer #1: smoke gate is **1 epoch + macro-AUROC > 0.65**.
* §10 answer #2: checkpoint format is `torch.save(dict)`; no safetensors.
* §10 answer #3: `_InMemoryTrainingDataset` caps at **20000 rows**;
  streaming variant deferred to v1.2.
* §10 answer #4: MPS-vs-CPU determinism accepted (intra-device only).
* §10 answer #5: default augmentations are `("hflip", "rotate10")`.
* §10 answer #6: TXRV-NIH fine-tuning **is not shipped** in v1.1; the
  trainer rejects any TXRV `backbone_id` with `ConfigError`.

**Code-level deviations from FINE_TUNING_DESIGN.md.**

* §3.1 / §5: the design doc declares
  `optimizer: Literal["adamw", "sgd"]` in §3.1 then walks it back to
  `Literal["adamw"]` in §5. The implementation ships **`Literal["adamw"]`
  only**; `__post_init__` rejects any other value with `ConfigError`. The
  `_ALLOWED_OPTIMIZERS` tuple in `harness/domain/types.py` is the single
  source of truth. v1.2 will lift the literal when SGD is wired.
* §5 (checkpoint resume): the design says "load the latest `epoch_*.pt`
  whose `config_hash` matches the current `TrainingConfig` (mismatch
  raises `AdapterError`)." Strict equality on the full config dict is
  incompatible with the design's own resume use case (§6.2 "train 3
  epochs to checkpoint dir, load via a fresh trainer with `n_epochs=5`").
  The implementation's `_config_hash` therefore **excludes** `n_epochs`
  and `early_stop_patience` from the hash via the
  `_HASH_EXCLUDED_FIELDS` constant; all other fields participate. A
  logically different run (different backbone, LR, augmentations, etc.)
  still fails fast with `AdapterError`.
* §5 (augmentation determinism): the design's open question between
  v2 transforms vs per-epoch global `torch.manual_seed` is resolved in
  favour of the v2 path. The trainer wraps each augmentation call in
  `torch.random.fork_rng(devices=[])` + a per-(epoch, batch)
  `torch.manual_seed(_augmentation_seed(base, epoch, batch_idx))`, so the
  augmentation pipeline is deterministic for a given `(seed, dataset)`
  pair without any global RNG mutation outside the `fork_rng` scope.
  ``_augmentation_seed`` derives the per-step seed via
  ``base_seed + epoch * 100003 + batch_idx`` so batch position 0 sees
  distinct augmentation parameters across epochs (Wave 4 review M1
  fix; the original ``base_seed + n_batches`` formulation reset
  ``n_batches`` per epoch and aliased batch 0 across epochs).
* §3.1 (`checkpoint_dir` typing): the design doc declares
  `checkpoint_dir: Path | None`; the v1.1 implementation initially
  shipped `str | None` (Wave 4 review M3). The fix wave updates the
  field to `Path | None` to match the design doc and CLAUDE.md's
  "pathlib.Path everywhere" rule. The factory call site no longer
  coerces to `str`; the trainer consumes the `Path` directly via
  ``ckpt_dir = config.checkpoint_dir`` and ``ckpt_dir / "epoch_NNN.pt"``.
* §3.2 (best-epoch weights): the design doc says "the returned
  ``TrainedClassifierPort`` corresponds to this [best] epoch's
  weights" and "URI of the persisted best-epoch checkpoint" but the
  v1.1 implementation initially returned a classifier wrapping the
  *last*-epoch weights, and `final_checkpoint_uri` pointed at
  ``epoch_{n_epochs_run-1:03d}.pt`` (Wave 4 review C1). The fix wave
  snapshots ``model.state_dict()`` whenever val-AUROC improves
  (`copy.deepcopy` into ``best_state``), restores it before
  constructing ``_TorchTrainedClassifier``, and rewrites
  `final_checkpoint_uri` to point at ``epoch_{best_epoch:03d}.pt``.
  Non-best per-epoch ``epoch_*.pt`` files remain on disk for resume.
* §13.9 model-card lineage (C2): the runner's first cut wired the
  trainer's port-level ``identifier`` (``"torch.finetune.v1"``) into
  both the ``backbone_id`` and ``head_id`` slots of the persisted
  ``ModelCard``, collapsing the densenet121-vs-resnet50 distinction
  (Wave 4 review C2). The fix wave routes ``config.backbone_id`` and
  ``config.head_id`` (set by the factory to
  ``"torch.finetune.{backbone}.v1"`` and ``"torch.finetune.linear.v1"``
  respectively) directly into ``_build_model_card``; the calibrator
  slot is unaffected (still uses ``_describe_port(calibrator, ...)``).
* §5 (`torch.load(weights_only=False)`): v1.1 uses
  `torch.load(weights_only=False)` to deserialise the full checkpoint
  dict including optimizer state, scheduler state, and numpy RNG
  state. PyTorch 2.4+ defaults to `weights_only=True`, which would
  reject this dict. Switching to `weights_only=True` requires
  splitting checkpoints into a `.pt` file (model + optimizer state
  dicts only) and a sidecar JSON file (config_hash, epoch counters,
  RNG state coerced to JSON). Deferred to v1.2 because (a) the
  harness's checkpoints are local-only artifacts in v1.1, and (b) the
  refactor touches the load/resume path with non-trivial regression
  risk this close to the v1.1 ship gate. Tracked as TODO for v1.2
  (Wave 4 review M4).
