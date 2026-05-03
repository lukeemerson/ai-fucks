# Fine-Tuning Architecture Design (v1.1)

**Status:** design proposal, awaiting user sign-off.
**Audience:** the implementation agent who will land the trainer adapter in a
follow-up PR; the reviewer who will adversarially read it.
**Scope:** end-to-end fine-tuning (backbone + head co-trained) on NIH-14, the
load-bearing experiment for the SOTA-chasing track.
**Outcome target:** macro-AUROC ~0.70 (frozen-feature baseline) → ~0.82-0.84
(CheXNet territory).

This document specifies the *port surface*, *adapter signature*, *composition
wiring*, *test strategy*, *determinism story*, and *out-of-scope items* for
fine-tuning. The Protocol stub ships in this PR; the adapter implementation,
factory wiring, and integration tests land in a follow-up PR by a separate
agent who reads this doc.

---

## 1. Why a new port (not an extension)

The existing pipeline is *frozen-feature* by design. ARCHITECTURE.md §1.2
("What `harness/` is **not**") says outright: "Not a training framework for
backbones. Backbone weights are *frozen* in v1." Section 9 ("Non-Goals for
v1") repeats the same claim under "Real torch training."

v1.1 changes that. The question is *how*. Three candidates were considered:

### 1.1 Option A — Extend `BackbonePort` with a `fit` method

Add `fit(...)` and a `predict_proba(...)` directly to `BackbonePort`. Every
existing adapter (`TorchVisionResNet50Backbone`, `TorchVisionDenseNet121Backbone`,
`TXRVDenseNet121NIHBackbone`, `IdentityFakeBackbone`, `CachedBackbone`) would
need a stub implementation that raises `AdapterError("not trainable")`.

**Rejected.** This pollutes the eval-only port with a method that 5/6
adapters cannot meaningfully implement. It also conflates feature extraction
(what `BackbonePort.extract` does) with *end-to-end multi-label
classification training* (which is fundamentally a different operation: it
needs labels, an optimizer, a loss, a head, and a DataLoader, none of which
the eval-only port knows about).

### 1.2 Option B — Subclass `TrainableBackbonePort(BackbonePort)`

Define a new Protocol that *extends* `BackbonePort` with `fit` and
`predict_proba`. Adapters that support training implement the subprotocol;
the runner type-narrows on `isinstance(backbone, TrainableBackbonePort)`.

**Rejected.** This still smuggles head + loss + optimizer + DataLoader
concerns into "the backbone." It also forces the trainable adapter to expose
both the `extract` semantics (image batch in, features out) *and* the
training loop, which then have to share state. The cleanest separation is to
treat the trainer as a separate concern.

### 1.3 Option C — A new `TrainerPort` that owns the training loop

Introduce a fresh port whose single job is to take an *un-fitted model* + a
*labeled dataset* and produce a *fitted model* that exposes a
`predict_proba(images) -> probabilities` method. The trainer owns the
optimizer, the LR schedule, the DataLoader, the augmentation pipeline, and
the checkpoint format. The fitted model satisfies a *new, narrow protocol*
(`TrainedClassifierPort`) that the runner can call in place of the
`BackbonePort.extract -> ClassifierHeadPort.predict_proba` chain.

**Adopted.** Clean hexagonal split:

* `BackbonePort` stays eval-only, unchanged. Existing adapters keep their
  signatures. The frozen-feature pipeline keeps working byte-for-byte.
* `TrainerPort` is the new training surface. Multiple adapters can implement
  it (`TorchFineTuneTrainer` first; `TorchLoraTrainer`, `TorchSwaTrainer`
  later if we want them).
* `TrainedClassifierPort` is what the trainer *returns*. It exposes a single
  method, `predict_proba(images: NDArray[np.float32]) -> NDArray[np.float32]`,
  shape-compatible with the existing eval pipeline so the runner can feed
  its output into `CalibratorPort` + `ThresholdPort` + `MetricsPort`
  unchanged.

This satisfies CLAUDE.md's YAGNI clause: we don't grow the existing
backbone/head ports until v1.2 actually needs the change. Frozen-feature
runs (the publication factory `build_publication_runner_v1`) keep their
existing wiring; fine-tune runs go through a new factory
(`build_finetune_runner_v1`, deferred to the implementation PR) that returns
a different `RunnerBundle` shape.

---

## 2. Port surface

Two new Protocols ship in `harness/ports/trainer.py`. Both are
`@runtime_checkable` per ARCHITECTURE.md §4.

### 2.1 `TrainedClassifierPort`

The thing the trainer produces. Exposes only `predict_proba` and
`embedding_dim` so the runner can route its output into the existing
calibrator/threshold/metrics chain without further reshape.

```python
@runtime_checkable
class TrainedClassifierPort(Protocol):
    """A model that has been trained end-to-end and is ready for inference.

    Returned by ``TrainerPort.fit``. The runner treats this as the
    backbone-plus-head replacement: instead of calling
    ``BackbonePort.extract`` then ``ClassifierHeadPort.predict_proba``, it
    calls ``TrainedClassifierPort.predict_proba`` directly on raw NHWC
    images.

    Implementations must be eval-only and pure functions of their inputs.
    Same image batch, same instance -> identical output.
    """

    @property
    def n_labels(self) -> int:
        """Number of output classes (== ``len(label_names)``)."""
        ...

    @property
    def identifier(self) -> str:
        """Stable string identifier (recorded in the model card)."""
        ...

    def predict_proba(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        """Map ``(N, H, W, C)`` images to ``(N, n_labels)`` probabilities.

        All output values must lie in ``[0, 1]``. Adapters raise
        :class:`AdapterError` on shape/dtype mismatch.
        """
        ...
```

### 2.2 `TrainerPort`

Takes a `TrainingDatasetPort` (described below) plus a `TrainingConfig`
(domain dataclass, described below) and returns a `TrainedClassifierPort`.

```python
@runtime_checkable
class TrainerPort(Protocol):
    """End-to-end model trainer.

    ``fit`` consumes a dataset that yields (image, labels) pairs and
    returns a trained model. The trainer owns the optimizer, the loss,
    the LR schedule, augmentation, and (where applicable) the
    checkpoint format. It does NOT own dataset construction or
    train/val splitting -- those flow in via ``training_dataset`` and
    ``validation_dataset`` keyword arguments.
    """

    @property
    def identifier(self) -> str:
        """Stable identifier for the training recipe (e.g. ``"torch.finetune.densenet121.v1"``)."""
        ...

    def fit(
        self,
        *,
        training_dataset: TrainingDatasetPort,
        validation_dataset: TrainingDatasetPort,
        config: TrainingConfig,
        seed: int,
    ) -> TrainedClassifierPort:
        """Train the model and return the eval-ready classifier.

        Implementations:

        * Must be deterministic given ``(training_dataset, validation_dataset,
          config, seed)``. Same inputs -> byte-identical model weights.
        * Must NOT touch global RNG state outside what is reachable via
          ``seed`` (no ``torch.manual_seed`` in module scope, no
          ``np.random.seed``). Use scoped ``torch.Generator`` / numpy
          ``Generator`` instances.
        * Must wrap third-party exceptions in :class:`AdapterError`.
        * Must respect ``config.checkpoint_dir`` for resume-from-checkpoint:
          on a second invocation with the same dir, training resumes from
          the latest checkpoint rather than restarting.
        """
        ...
```

### 2.3 `TrainingDatasetPort`

The trainer needs an iterable that yields `(image, labels)` rows for an
arbitrary number of epochs. The existing `DatasetPort` is shape-incompatible
(it returns a frozen `Dataset` snapshot of `Sample`s with `image_ref`s and
no in-memory image data). Two options were considered:

* **A.** Reuse `DatasetPort` and require the trainer adapter to construct a
  `torch.utils.data.Dataset` internally, calling
  `dataset.get_image_bytes(sample.image_ref)` per row. Forces the trainer
  to know about bytes encoding.
* **B.** Define a narrow `TrainingDatasetPort` whose `__iter__` /
  `__len__` / `__getitem__` deliver decoded `(NDArray[np.float32], NDArray[np.int8])`
  rows directly. The composition root builds this from the existing
  `DatasetPort` + a split's index list.

**Adopted: B.** Trainer adapters speak in terms of decoded numpy rows; the
composition root handles the byte-to-tensor decoding (same pattern as the
publication factory's `_DecodingDataset` wrapper).

```python
@runtime_checkable
class TrainingDatasetPort(Protocol):
    """A finite-length iterable of decoded ``(image, labels)`` rows.

    Construction is the composition root's responsibility: it wraps the
    existing :class:`DatasetPort` plus a split's index list into a thin
    in-memory adapter. The trainer reads images and labels via
    ``__getitem__`` and ``__len__`` (PyTorch's ``DataLoader``-friendly
    surface) without ever seeing raw bytes or ``Sample`` records.
    """

    def __len__(self) -> int:
        """Number of rows in this view."""
        ...

    def __getitem__(
        self, index: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
        """Return ``(image, labels)`` for row ``index``.

        ``image`` is shape ``(H, W, C)`` float32 in ``[0, 1]``; ``labels`` is
        shape ``(K,)`` int8 multi-hot.
        """
        ...
```

`TrainingDatasetPort` is intentionally minimal -- no `__iter__` (PyTorch's
`DataLoader` derives one from `__len__` + `__getitem__`); no metadata
accessors (the trainer doesn't need them); no patient ID surface (splitting
already happened upstream).

---

## 3. Domain types

Two new frozen dataclasses ship in `harness/domain/types.py` (added in the
implementation PR). They mirror the existing `BootstrapConfig` /
`ThresholdConfig` style: pure data, `__post_init__` validation,
`@dataclass(frozen=True, slots=True)`.

### 3.1 `TrainingConfig`

```python
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for a single fine-tuning run.

    All fields are explicit: no defaults, no None placeholders. The
    composition root assembles a ``TrainingConfig`` from
    ``ExperimentConfig.training`` (a new optional sub-config); the trainer
    adapter consumes it verbatim.
    """

    backbone_id: str
    """Stable backbone identifier (e.g. ``"torchvision.densenet121"``).
    The trainer adapter looks this up and constructs the matching
    ``nn.Module``; unknown identifiers raise ``ConfigError``."""

    n_labels: int
    """Number of output labels (must equal ``len(ExperimentConfig.label_names)``)."""

    n_epochs: int
    """Number of full passes over ``training_dataset``. Must be positive."""

    batch_size: int
    """Per-step batch size for both training and validation. Must be positive."""

    learning_rate: float
    """Initial LR for the optimizer. Must be > 0."""

    weight_decay: float
    """L2 regularization. Must be >= 0."""

    optimizer: Literal["adamw"]
    """Optimizer choice. v1.1 ships ``"adamw"`` only; the literal is
    narrowed to ``Literal["adamw"]`` to remove the §3 / §5 drift the
    original design surfaced. The ``__post_init__`` validator rejects
    any other value with ``ConfigError``. v1.2 will lift the literal
    when SGD is wired (see §3.1 deviation, this section, in the
    implementation PR)."""

    lr_schedule: Literal["cosine", "constant"]
    """LR schedule. v1.1 ships both; defaults to ``"cosine"`` per CheXNet's recipe."""

    warmup_epochs: int
    """Number of warmup epochs for cosine schedule. Must be >= 0 and <= n_epochs."""

    augmentations: tuple[str, ...]
    """Ordered list of augmentation names (e.g. ``("hflip", "rotate10", "colorjitter")``).
    The trainer adapter resolves each name to a torchvision transform; unknown
    names raise ``ConfigError``. Empty tuple means no augmentation."""

    image_size: tuple[int, int]
    """Trainer input ``(H, W)`` after the dataset's resize. Must match what
    the loaded backbone expects (224x224 for DenseNet121/ResNet50)."""

    checkpoint_dir: Path | None
    """If set, the trainer writes ``epoch_<n>.pt`` checkpoints here and
    resumes from the latest on a re-run. ``None`` disables checkpointing
    (training restarts every call)."""

    early_stop_patience: int | None
    """If set, stop training after this many epochs without val-AUROC
    improvement. ``None`` disables early stopping (run all ``n_epochs``).
    Must be > 0 when set."""

    num_dataloader_workers: int
    """``torch.utils.data.DataLoader.num_workers``. v1.1 enforces ``0``
    (single-process) per ARCHITECTURE.md §9 ("Multi-process data loading"
    is non-goal); kept on the config so v1.2 can lift it."""
```

`__post_init__` validation:

* `n_labels`, `n_epochs`, `batch_size` > 0.
* `learning_rate` > 0; `weight_decay` >= 0.
* `optimizer in {"adamw", "sgd"}` (raises `ConfigError`).
* `lr_schedule in {"cosine", "constant"}`.
* `warmup_epochs >= 0`; `warmup_epochs <= n_epochs`.
* `image_size[0] > 0`, `image_size[1] > 0`.
* `early_stop_patience is None or early_stop_patience > 0`.
* `num_dataloader_workers == 0` (raise `ConfigError` otherwise; v1.1 lock).

#### §3.1 implementation deviation

The original §3.1 declaration `optimizer: Literal["adamw", "sgd"]` was
walked back inline in §5 ("Wait — that is YAGNI-violating. Revised: drop
``"sgd"`` from the Literal until v1.2"). The implementation PR ships the
narrower `Literal["adamw"]` form everywhere -- in the docstring above
and in the `_ALLOWED_OPTIMIZERS` tuple in `harness/domain/types.py` --
so both the doc and the code agree. v1.2 will lift the literal when
SGD is wired.

### 3.2 `TrainingResult`

```python
@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Bookkeeping returned alongside the ``TrainedClassifierPort``.

    Persisted into the model card's ``notes`` field; not consumed by the
    runner's downstream calibrator/threshold/metrics chain.
    """

    n_epochs_run: int
    """Actual number of epochs executed (may be < config.n_epochs if
    early stopping fired)."""

    train_loss_per_epoch: tuple[float, ...]
    """Mean training BCE loss per epoch."""

    val_loss_per_epoch: tuple[float, ...]
    """Mean validation BCE loss per epoch."""

    val_macro_auroc_per_epoch: tuple[float, ...]
    """Macro-AUROC on the validation set per epoch (early stop signal)."""

    best_epoch: int
    """0-indexed epoch at which val_macro_auroc was maximal. The returned
    ``TrainedClassifierPort`` corresponds to this epoch's weights."""

    final_checkpoint_uri: str | None
    """URI of the persisted best-epoch checkpoint, or ``None`` if no
    checkpoint dir was configured."""
```

`TrainerPort.fit` returns `tuple[TrainedClassifierPort, TrainingResult]`.
The runner records the result in the model card; the trained classifier
flows into the eval chain.

#### §3.2 fix-wave deviation (best-epoch weights restoration)

The Wave 4 review (C1) flagged that the v1.1 trainer's first cut
returned a `_TorchTrainedClassifier` wrapping the *last*-epoch model
state and computed `final_checkpoint_uri` as
``epoch_{n_epochs_run-1:03d}.pt``, both contradicting the design's
"corresponds to this [best] epoch's weights" / "URI of the persisted
best-epoch checkpoint" wording. The fix wave snapshots
`model.state_dict()` whenever val-AUROC improves
(`copy.deepcopy(...)` into `best_state`), restores it before
constructing the returned classifier, and rewrites
`final_checkpoint_uri` to point at ``epoch_{best_epoch:03d}.pt``.
On resume the wave seeds `best_state` from the existing
``epoch_{best_epoch_so_far:03d}.pt`` so a pure-resume call (no new
epochs) still returns best-epoch weights. Non-best per-epoch
``epoch_*.pt`` files remain on disk for future resume.

---

## 4. Composition wiring

Fine-tuning replaces both `BackbonePort.extract` *and* `ClassifierHeadPort.fit/predict_proba`
in the existing pipeline. Two wiring approaches:

### 4.1 Option A — Add a `fine_tune` flag to `run_experiment`

```python
def run_experiment(
    config: ExperimentConfig,
    *,
    dataset: DatasetPort,
    splitter: SplitterPort,
    backbone: BackbonePort | None,
    head: ClassifierHeadPort | None,
    trainer: TrainerPort | None,
    ...
) -> ExperimentResult:
    if trainer is not None:
        # fine-tune path
    else:
        # frozen-feature path
```

**Rejected.** Optional ports + `if/else` branches in the runner is exactly
the pattern ARCHITECTURE.md §6.1 explicitly forbids: "All ports are
keyword-only. No defaults — composition decides what to inject." Adding
optionals reverses that contract and makes the runner's signature lie about
what it requires.

### 4.2 Option B — A separate `run_finetune_experiment` function

```python
def run_finetune_experiment(
    config: ExperimentConfig,
    *,
    dataset: DatasetPort,
    splitter: SplitterPort,
    trainer: TrainerPort,
    calibrator: CalibratorPort,
    thresholds: ThresholdPort,
    metrics: MetricsPort,
    store: ArtifactStorePort,
    randomness: RandomnessPort,
    clock: datetime | None = None,
) -> ExperimentResult:
    ...
```

**Adopted.** Two named entry points, two clean signatures, two
non-overlapping `RunnerBundle` shapes (the existing `RunnerBundle` for
frozen-feature; a new `FineTuneRunnerBundle` for fine-tune that drops
`backbone` + `head` and adds `trainer`). No optional ports, no `if/else`.

The fine-tune runner's algorithm:

```
1.  randomness.seed_all(config.seed)
2.  ds = dataset.load(); validate label_names
3.  split_seed   = randomness.child_seed(config.seed, "split")
    trainer_seed = randomness.child_seed(config.seed, "trainer")
4.  split = splitter.split(ds, val_fraction, test_fraction, split_seed)
5.  train_dataset = _build_training_dataset(ds, split.train_indices, dataset)
    val_dataset   = _build_training_dataset(ds, split.val_indices, dataset)
    test_dataset  = _build_training_dataset(ds, split.test_indices, dataset)
6.  trained, training_result = trainer.fit(
        training_dataset=train_dataset,
        validation_dataset=val_dataset,
        config=config.training,
        seed=trainer_seed,
    )
7.  val_raw  = trained.predict_proba(_images_array(val_dataset))
    test_raw = trained.predict_proba(_images_array(test_dataset))
8.  calibrator.fit(val_raw, val_labels)
    val_calibrated  = calibrator.transform(val_raw)
    test_calibrated = calibrator.transform(test_raw)
9.  threshold_set = thresholds.fit(val_calibrated, val_labels, config=config.threshold)
    test_predictions = thresholds.apply(test_calibrated, threshold_set)
    report = metrics.evaluate(test_calibrated, test_labels, threshold_set,
                              bootstrap=config.bootstrap)
10. model_card = _build_model_card(..., trainer_id=trainer.identifier,
                                   training_result=training_result)
11. persist artifacts (incl. checkpoint URI in the model card notes)
12. return ExperimentResult(...)
```

Steps 8-12 are byte-identical to the frozen-feature runner. The fork is
strictly between steps 6-7 (where features-then-fit becomes train-then-eval).
A cautious refactor in the implementation PR may extract steps 8-12 into a
shared helper; the design does not depend on that.

### 4.3 `_build_training_dataset` helper

```python
def _build_training_dataset(
    ds: Dataset,
    indices: Sequence[int],
    dataset: DatasetPort,
) -> TrainingDatasetPort:
    """Wrap (Dataset, indices, byte-source) into a TrainingDatasetPort."""
    ...
```

Implemented in `harness/composition/_finetune_pipeline.py` as
`_InMemoryTrainingDataset`, which decodes each `image_ref` once into a
`(H, W, C) float32 [0, 1]` row and returns it from `__getitem__`. v1.1 is
in-memory; v1.2 may add a streaming variant (`_StreamingTrainingDataset`)
that decodes lazily. Decoding uses the existing `DatasetPort.get_image_bytes`
plus the `NIHImageLoader.decode` helper (the same one
`_DecodingDataset` uses for the frozen-feature path).

### 4.4 `ExperimentConfig` extension

Add a new optional sub-config to `ExperimentConfig`:

```python
@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    ...
    training: TrainingConfig | None  # None for frozen-feature runs
```

`run_finetune_experiment` raises `ConfigError` if `config.training is None`;
`run_experiment` (the existing frozen-feature runner) raises `ConfigError`
if `config.training is not None`. Mutually exclusive, validated up front.

### 4.5 New factory (deferred to implementation PR)

`build_finetune_runner_v1(seed, *, nih_csv_path, nih_images_dir,
artifact_root, training_config, head_choice, ...)` returns a
`FineTuneRunnerBundle`:

```python
@dataclass(frozen=True, slots=True)
class FineTuneRunnerBundle:
    config: ExperimentConfig
    dataset: DatasetPort
    splitter: SplitterPort
    trainer: TrainerPort
    calibrator: CalibratorPort
    thresholds: ThresholdPort
    metrics: MetricsPort
    store: ArtifactStorePort
    randomness: RandomnessPort
```

Not implemented in this design PR; documented here so the implementation
agent has a target.

---

## 5. Adapter design (`harness/adapters/torch/trainer.py`)

The first adapter shipped is `TorchFineTuneTrainer` (in the implementation
PR). Spec:

* **Backbones supported.** v1.1 ships `densenet121` and `resnet50` only,
  matching the existing eval-only adapters. The adapter resolves
  `config.backbone_id` to a torchvision constructor (the same lookup the
  eval-only `TorchVisionDenseNet121Backbone` uses internally) and replaces
  the classifier head with `nn.Linear(in_features, config.n_labels)`.
* **Loss.** `nn.BCEWithLogitsLoss` -- multi-label BCE on logits, no
  softmax. v1.1 ships only this loss; the design intentionally does NOT
  expose a `loss=` parameter on `TrainingConfig` (per CLAUDE.md YAGNI;
  documented in §8 below).
* **Optimizer.** `torch.optim.AdamW(model.parameters(), lr, weight_decay)`.
  SGD is plumbed through `TrainingConfig.optimizer` as a future-proofing
  literal but the initial adapter raises `AdapterError` if `optimizer=="sgd"`
  (deferred). Wait — that is YAGNI-violating. Revised: drop `"sgd"` from the
  Literal until v1.2; the field is `Literal["adamw"]` only in v1.1, lifted
  to `Literal["adamw", "sgd"]` when SGD ships.
* **LR schedule.** `torch.optim.lr_scheduler.CosineAnnealingLR` with
  `warmup_epochs` of linear warmup before the cosine, or
  `lr_scheduler.ConstantLR` for `"constant"`.
* **DataLoader.** `torch.utils.data.DataLoader(training_dataset,
  batch_size=config.batch_size, shuffle=True, num_workers=0,
  generator=torch.Generator().manual_seed(seed_for_dataloader))`. The
  trainer constructs the loader internally; the composition root passes a
  `TrainingDatasetPort`, not a pre-built loader. This keeps DataLoader
  config inside the adapter where it belongs.
* **Augmentation.** Each name in `config.augmentations` maps to a
  torchvision transform (`hflip` -> `RandomHorizontalFlip(p=0.5)`,
  `rotate10` -> `RandomRotation(degrees=10)`, etc.). The list is composed
  into a single `transforms.Compose` and applied at DataLoader time.
  Augmentations are training-only; the validation loader uses no augmentation.
  Augmentation seeding goes through a `torch.Generator` derived from
  `seed`, not global RNG.
* **Device.** Same auto-fallback ladder as the eval-only adapters
  (`mps -> cuda -> cpu`). Caller override via a future `device=` constructor
  argument; v1.1 always uses auto. MPS is the practical target on the
  user's M-series Mac.
* **Mixed precision.** Out of scope for v1.1 (see §8).
* **Checkpointing.** On every epoch end, save
  `{checkpoint_dir}/epoch_{n:03d}.pt` containing:

  ```python
  {
      "epoch": int,
      "model_state_dict": dict,
      "optimizer_state_dict": dict,
      "scheduler_state_dict": dict,
      "torch_rng_state": tensor,
      "numpy_rng_state": dict,
      "config_hash": str,
      "best_val_macro_auroc_so_far": float,
      "epochs_without_improvement": int,
  }
  ```

  On resume: load the latest `epoch_*.pt` whose `config_hash` matches the
  current `TrainingConfig` (mismatch raises `AdapterError`), restore all
  state, continue from `epoch + 1`. Mismatched config means a logically
  different run; the operator must use a different `checkpoint_dir`.

  **§5 implementation deviation: hash excludes `n_epochs` and
  `early_stop_patience`.** Strict equality on the full config dict is
  incompatible with the design's own resume use case in §6.2 ("train 3
  epochs to checkpoint dir, load via a fresh trainer with `n_epochs=5`,
  assert it ran exactly 2 more epochs"). The implementation's
  `_config_hash` excludes those two fields via the
  `_HASH_EXCLUDED_FIELDS` constant in `harness/adapters/torch/trainer.py`;
  every other field participates. A logically different run (different
  backbone, LR, augmentations, image size, etc.) still fails fast with
  `AdapterError`.

* **Returned `TrainedClassifierPort`.** A small adapter class
  (`_TorchTrainedClassifier`) that wraps the trained `nn.Module` in
  `eval()` mode and exposes `predict_proba` as
  `sigmoid(model(images_to_tensor(images)))`. Eval-only,
  `torch.no_grad()`-wrapped, `.detach().cpu().numpy()` at the boundary
  (mirrors the eval-only backbone adapters).

* **Determinism.** No global `torch.manual_seed`. The trainer constructs
  its own `torch.Generator` seeded from `seed` and threads it through:
  - DataLoader's `generator=` argument.
  - `torch.utils.data.DataLoader(worker_init_fn=...)` (no-op for
    `num_workers=0` but documented for v1.2).
  - Each `torchvision.transforms` augmentation that accepts a `generator=`
    argument.

  When the chosen device is CUDA, the constructor sets
  `torch.backends.cudnn.deterministic=True` and `cudnn.benchmark=False`.
  MPS does not provide an equivalent flag; runs on MPS are documented as
  non-byte-reproducible across devices (same v1 limitation as the eval-only
  adapters; see ARCHITECTURE.md §13.5).

  **v1.1 deviation (open question).** Some torchvision transforms
  (`RandomRotation`, `ColorJitter`) sample from the global torch RNG and
  do not currently accept a generator argument. If the implementation agent
  finds this a blocker, two fall-back strategies in priority order:

  1. Use `torchvision.transforms.v2` which is generator-aware end-to-end.
  2. Fall back to `torch.manual_seed(seed)` once at the start of each
     epoch and document the deviation in §13 of ARCHITECTURE.md (same
     pattern the eval-only torch backbones use).

  Decision deferred to the implementation agent.

---

## 6. Test strategy

### 6.1 Contract tests (`tests/harness/contract/test_trainer_contract.py`)

A single base class `TrainerPortContract` asserts the universal port
behaviors. Subclassed once per real adapter (none in this PR; the file
contains the abstract base only).

Required assertions:

* `test_fit_returns_trained_classifier_port`: shape & type of return.
* `test_predict_proba_outputs_in_unit_interval`: every value in `[0, 1]`,
  shape `(N, n_labels)`.
* `test_predict_proba_is_eval_only`: two consecutive calls with identical
  inputs return byte-identical outputs (no dropout, no batchnorm drift).
* `test_loss_decreases_monotonically`: on a tiny synthetic dataset (16
  rows, 2 classes, perfectly separable), the trainer drives
  `train_loss_per_epoch` to decrease across all epochs (allowing one or
  two epochs of plateau via `np.diff(losses).max() < small_tolerance`).
* `test_determinism_byte_identical_weights`: same seed + same config + same
  dataset -> byte-identical `state_dict()` hashes (or, for adapters that
  cannot expose state_dict, byte-identical `predict_proba` outputs on a
  fixed eval batch).
* `test_fit_rejects_mismatched_n_labels`: when `dataset[i][1].shape[0] !=
  config.n_labels`, raise `AdapterError`.
* `test_fit_rejects_empty_dataset`: `len(training_dataset) == 0` raises
  `AdapterError`.
* `test_invalid_optimizer_raises_config_error`: covered at the
  `TrainingConfig.__post_init__` level (`ConfigError`), not the adapter,
  but the contract asserts this funnels through `HarnessError`.

A helper `class _TinyDataset(TrainingDatasetPort)` lives in the contract
test file -- 16 rows, 8x8 grayscale, two-class synthetic where class 0 is
"dark image" and class 1 is "bright image." Solvable in 2-3 epochs. Used
by every contract subclass.

The base class has no `adapter` fixture default; subclasses must override.
The implementation PR adds `class TestTorchFineTuneTrainerContract`; this
PR ships only the abstract base.

### 6.2 Unit tests (deferred to implementation PR)

In `tests/harness/unit/adapters/torch/test_finetune_trainer.py`:

* `test_loss_is_bce_with_logits`: assert the adapter calls
  `nn.BCEWithLogitsLoss` (introspect via attribute, not by patching).
* `test_optimizer_is_adamw_when_configured`: ditto for `torch.optim.AdamW`.
* `test_cosine_schedule_warmup_then_decay`: instrument
  `optimizer.param_groups[0]["lr"]` over a 10-epoch fake run and assert
  it climbs linearly through `warmup_epochs`, then declines per cosine.
* `test_checkpoint_resumes_correctly`: train 3 epochs to checkpoint dir,
  load via a fresh trainer with `n_epochs=5`, assert it ran exactly 2
  more epochs and the final `state_dict` equals what 5 epochs of fresh
  training would produce.
* `test_augmentation_pipeline_seeded`: same `seed` + same input batch
  produces the same augmented images byte-for-byte.
* `test_validation_loader_has_no_augmentation`: snapshot-test that the
  val DataLoader passes images through unchanged.

All marked `@pytest.mark.torch` (excluded from default run).

### 6.3 Integration test (`tests/harness/integration/test_finetune_runner_smoke.py`, deferred)

* Build a fine-tune runner via `build_finetune_runner_v1(seed=0,
  nih_csv_path=fixture, nih_images_dir=fixture, artifact_root=tmp_path,
  training_config=TrainingConfig(n_epochs=2, batch_size=4, ...))`.
* Run on the existing 16-row NIH fixture.
* Assert `ExperimentResult` shape: every field populated, model card
  records the trainer identifier and `n_epochs_run=2`.
* Assert `report.macro_auroc.point` is finite and in `(0, 1)`. (Don't
  assert a quality bar -- 16 rows is too small for that.)

Marked `@pytest.mark.torch` and `@pytest.mark.slow` (training even 2
epochs on 16 rows takes >5s in practice).

### 6.4 Smoke test (`@pytest.mark.smoke`, deferred to implementation PR)

* Build the fine-tune runner against the real 4999-sample NIH slice.
* Train for 1 epoch only.
* Assert `report.macro_auroc.point > 0.65` -- the frozen-feature floor
  the existing pipeline achieved on this slice. One epoch of fine-tuning
  on 4k samples should at minimum match this; failure to clear it
  signals a regression in the trainer adapter.

This is the single load-bearing quality gate. If the implementation PR's
smoke test fails, the trainer is buggy and the PR is rejected.

---

## 7. Determinism

The fine-tuning surface multiplies the determinism risk surface
substantially. Concrete rules, all enforced by contract tests:

* **No global RNG mutation.** No `torch.manual_seed`, `np.random.seed`, or
  `random.seed` at module scope or in the trainer's `__init__` outside of
  `torch.Generator` instances scoped to that call.
* **Sub-seed derivation.** The runner derives `trainer_seed =
  randomness.child_seed(config.seed, "trainer")` and passes it to
  `TrainerPort.fit(seed=...)`. The trainer in turn derives
  `dataloader_seed = derive_subseed(seed, "dataloader")`,
  `aug_seed = derive_subseed(seed, "augment")`,
  `init_seed = derive_subseed(seed, "init")` for layer initialization
  (only relevant when `weights=None`, which fine-tune runs never use).
  Sub-seeding within the adapter uses the same SHA-256-based
  `child_seed` algorithm as the runner; the implementation extracts that
  helper to `harness/domain/seed.py` so trainer + runner share the
  derivation.
* **Checkpoint state.** Every checkpoint records the torch + numpy RNG
  state at end-of-epoch. On resume, both states restore exactly;
  resume-from-epoch-N produces byte-identical weights to a fresh run that
  ran N+M epochs.
* **DataLoader determinism.** `shuffle=True` is fine when seeded via
  `generator=` argument; `worker_init_fn` is documented but no-op at
  `num_workers=0`. Multi-worker DataLoaders are explicitly out of scope
  for v1.1 (per ARCHITECTURE.md §9).
* **MPS caveat.** Same as the eval-only adapters: MPS does not provide a
  `cudnn.deterministic`-equivalent flag. Runs on MPS are *intra-device*
  reproducible (same Mac, same torch build, same seed -> same weights)
  but not byte-equal to runs on CPU or CUDA. Documented in the model
  card's `notes` field.

---

## 8. Out of scope (explicit YAGNI list)

Per CLAUDE.md's YAGNI clause: don't add fields "we might need."
Documented as deferred so v1.2 inherits a coherent backlog rather than a
half-baked surface.

* **Multi-GPU / DDP / FSDP.** One MPS device. v1.2 if we ever rent GPUs.
* **Mixed precision (FP16 / BF16).** MPS supports BF16 for some ops but
  not all; introducing autocast adds a determinism risk surface that
  isn't justified by our tiny model. v1.2 follow-up if training time
  becomes the bottleneck.
* **Hyperparameter search.** Manual ablation rows for v1.1 -- one
  `TrainingConfig` per ablation, fed through the existing seed-grid
  ablation runner (`harness/scripts/run_ablation.py`). A `SearchPort` is
  v1.2.
* **Wandb / mlflow / TensorBoard.** Logs go to disk via the existing
  `ArtifactStorePort`; no third-party experiment trackers. The training
  curves shipped in `TrainingResult` are the *only* logged surface.
* **Multi-dataset training (CheXpert + MIMIC).** Gated on credentialing
  (out per ARCHITECTURE.md §1.2). v1.2 if the credentialing path
  changes.
* **Augmentation libraries (`albumentations`, `kornia`).**
  `torchvision.transforms.v2` covers our needs (hflip, rotate, color
  jitter, gaussian blur, random crop) at zero new dependency cost.
* **Other backbones (`timm` ConvNeXt, ViT, EVA, Swin).** v1.2. v1.1
  sticks with proven DenseNet121 / ResNet50.
* **TXRV-DenseNet121 fine-tuning.** Adding the TXRV
  ("densenet121-res224-nih") backbone to the trainer is one extra
  branch in the backbone-resolution lookup; *technically* trivial but
  CLAUDE.md (and the embedding ablation paper-bait note) calls out that
  TXRV NIH weights were trained on rows we evaluate on. Fine-tuning
  TXRV on top of those leaked weights is double-leaky. v1.2 is the
  earliest sensible time, and only with the leakage explicitly
  documented.
* **Pseudolabeling, knowledge distillation, SWA, EMA.** v1.2+.
* **Loss function variants (focal, asymmetric, distribution-balanced).**
  v1.1 ships only `nn.BCEWithLogitsLoss`. The `TrainingConfig` does not
  expose a `loss=` parameter. Reasoning: focal/asymmetric loss is the
  third-most-impactful lever after backbone + augmentations on multi-label
  CXR; landing them as a follow-up keeps the v1.1 BCE baseline cleanly
  separable from the loss ablation. v1.2 lifts the literal.
* **Class-weighted BCE (`pos_weight`).** Same reasoning as focal/asymmetric.
  Even though it's a one-line change, exposing it now means designing the
  weight-derivation rule (per-class prevalence inverse? sqrt? capped?), and
  the right answer depends on the v1.1 BCE baseline's per-class behavior.
  v1.2.
* **DataLoader multi-worker (`num_workers > 0`).** ARCHITECTURE.md §9.
  The `TrainingConfig.num_dataloader_workers` field exists and is
  validated to `0` so v1.2 can lift the constraint without a config
  schema change.
* **Streaming / memmap-backed `TrainingDatasetPort`.** v1.1 is in-memory
  only. The full NIH-14 fits comfortably (~112k * 224 * 224 * 1 * 4
  bytes ~= 22 GB; tight on a 32 GB Mac with the OS + browser + torch
  also resident, *very* tight on a 16 GB Mac). The implementation PR
  may choose to chunk-load if memory is a problem, but the port surface
  is stable.
* **Ensembling across seeds.** v1.2.

---

## 9. File layout

| File | Purpose | Ships in this PR? |
| ---- | ------- | ----------------- |
| `harness/docs/FINE_TUNING_DESIGN.md` | This document | YES |
| `harness/ports/trainer.py` | `TrainerPort`, `TrainedClassifierPort`, `TrainingDatasetPort` Protocol stubs | YES |
| `harness/docs/ARCHITECTURE.md` | §13 update pointing at this doc | YES (small section bump) |
| `tests/harness/contract/test_trainer_contract.py` | Abstract `TrainerPortContract` base class, no concrete subclasses yet | YES |
| `harness/domain/types.py` | `TrainingConfig`, `TrainingResult`, `ExperimentConfig.training` | NO (impl PR) |
| `harness/adapters/torch/trainer.py` | `TorchFineTuneTrainer` adapter | NO (impl PR) |
| `harness/composition/runner.py` | `run_finetune_experiment` | NO (impl PR) |
| `harness/composition/_finetune_pipeline.py` | `_InMemoryTrainingDataset` helper, `FineTuneRunnerBundle` dataclass | NO (impl PR) |
| `harness/composition/factories.py` | `build_finetune_runner_v1` factory | NO (impl PR) |
| `tests/harness/unit/adapters/torch/test_finetune_trainer.py` | Unit tests | NO (impl PR) |
| `tests/harness/integration/test_finetune_runner_smoke.py` | Integration test on 16-row fixture | NO (impl PR) |
| Smoke test against 4999-row slice | macro-AUROC > 0.65 gate | NO (impl PR) |

---

## 10. Open questions for user sign-off

These are decisions worth flagging explicitly before the implementation
agent starts. Default answers below; the user may override.

1. **Single epoch budget for the smoke quality gate (§6.4).** Smoke tests
   should be machine-dependent but bounded. 1 epoch on 4999 rows at the
   user's MPS throughput is roughly 2-3 minutes; that is at the edge of
   acceptable for `pytest -m smoke`. Alternatives:
   * Cap at 1 epoch + macro-AUROC > 0.65 (default proposal).
   * 3 epochs + macro-AUROC > 0.70 (more discriminating signal but 3x
     wall-clock).
   * Skip the smoke gate entirely; rely only on the 16-row integration
     test for CI signal and run the 4999-row evaluation manually.

2. **Checkpoint format.** Default proposal is `torch.save(dict, path)` to
   `epoch_NNN.pt`. Alternatives:
   * `torch.save(model.state_dict(), path)` (slimmer, but loses optimizer
     and RNG state -- breaks resume determinism).
   * `safetensors` (cross-framework, no `pickle` payload, but adds a
     dependency for very little benefit at our scale).

3. **`TrainingDatasetPort` is in-memory only in v1.1.** Acceptable risk
   on a 32 GB Mac for the 4999-row slice and 14-class subset of the full
   ~112k corpus. If the user plans to fine-tune on the *full* 112k corpus
   on a 16 GB machine, this needs a streaming variant landing alongside
   the trainer adapter. (Not landing in this design PR either way.)

4. **MPS-vs-CPU determinism.** Documented as a known limitation
   (intra-device reproducible, not byte-equal across devices). User
   acceptance assumed; raise a flag if you want the smoke test pinned to
   CPU for byte-stability at the cost of training time.

5. **Augmentation default set.** Default proposal is
   `("hflip", "rotate10")` -- the CheXNet recipe. The brief doesn't pin
   this; the implementation PR can ship the default in `_make_finetune_config`
   and downstream ablations can override.

6. **Does fine-tuning with TXRV-NIH-pretrained DenseNet121 ship in
   v1.1?** Default: NO. Same leakage concern that applies to the eval-only
   TXRV backbone applies tenfold to fine-tuning on top of leaked weights.
   Worth a separate user decision; the design accommodates either choice
   (it would be one extra `backbone_id` lookup branch).

---

## 11. Path to implementation

The follow-up PR by a separate agent should:

1. Read this doc end-to-end. Cite the section number for any deviation.
2. Add `TrainingConfig` + `TrainingResult` to `harness/domain/types.py`
   with `__post_init__` validators (TDD: write the validator tests
   first).
3. Add the `training: TrainingConfig | None` field to `ExperimentConfig`.
4. Implement `_InMemoryTrainingDataset` in
   `harness/composition/_finetune_pipeline.py`. TDD: contract tests for
   `TrainingDatasetPort` (added to this design PR's contract test file
   as the abstract base) drive the implementation.
5. Implement `TorchFineTuneTrainer` adapter. TDD: subclass the
   `TrainerPortContract` base for the concrete adapter, watch the
   contract tests fail RED, implement to GREEN.
6. Add unit tests per §6.2.
7. Implement `run_finetune_experiment` and the
   `build_finetune_runner_v1` factory.
8. Add the integration test (§6.3) and smoke test (§6.4) markers.
9. Update ARCHITECTURE.md §13 with the v1.1 fine-tuning surface (new
   subsection, mirroring §13.5 / §13.6 / §13.7 in style).

Quality gates per CLAUDE.md: `pytest tests/`, `pytest tests/harness/ -m
smoke` (the new smoke test is the load-bearing one), `pytest tests/ -m
slow`, `ruff check`, `mypy --strict`. All clean.

---

## 12. Source-of-truth contract

This document is the source of truth for v1.1 fine-tuning. If the
implementation deviates from it, either (a) update the implementation, or
(b) update this document in the same PR with a §-numbered deviation
entry. Do not silently let the doc rot.

End of design.
