"""Factories that construct concrete adapter sets for ``run_experiment``.

Each factory returns a :class:`RunnerBundle` -- a frozen dataclass with one
explicitly-typed field per port plus a ready-to-run :class:`ExperimentConfig`.
The bundle can be unpacked directly into
:func:`harness.composition.runner.run_experiment` without ``cast`` calls.

These factories are the only place adapters are instantiated outside tests --
production CLI / notebook entry points should call one of these and pass the
result into the runner verbatim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from harness.adapters.fakes.artifact_store import InMemoryFakeArtifactStore
from harness.adapters.fakes.backbone import IdentityFakeBackbone
from harness.adapters.fakes.calibrator import IdentityFakeCalibrator
from harness.adapters.fakes.classifier_head import LinearFakeClassifierHead
from harness.adapters.fakes.dataset import InMemoryFakeDataset
from harness.adapters.fakes.metrics import CountingFakeMetrics
from harness.adapters.fakes.randomness import SeededRandomness
from harness.adapters.fakes.splitter import DeterministicFakeSplitter
from harness.adapters.fakes.threshold import FixedFakeThreshold
from harness.adapters.fs.artifact_store import FilesystemArtifactStore
from harness.adapters.fs.cached_backbone import CachedBackbone
from harness.adapters.fs.nih_csv import NIH14_LABELS
from harness.adapters.fs.nih_dataset import NIHDataset, NIHDatasetConfig
from harness.adapters.sklearn.calibrator import (
    PerClassIsotonicCalibrator,
    PerClassPlattCalibrator,
)
from harness.adapters.sklearn.head import SklearnGradientBoostingHead
from harness.adapters.sklearn.lr_head import SklearnLogisticRegressionHead
from harness.adapters.sklearn.metrics import BootstrapMetrics
from harness.adapters.sklearn.splitter import IterativeStratifiedPatientSplitter
from harness.adapters.sklearn.threshold import PrSweepShrinkageThreshold
from harness.composition._finetune_pipeline import (
    FineTuneRunnerBundle,
    ImageDecoder,
)
from harness.composition._publication_pipeline import (
    DecodedImageCache,
    _DecodingBackbone,
    _DecodingDataset,
    _SubsettedDataset,
)
from harness.composition.runner import BYTES_IMAGE_SHAPE
from harness.domain.errors import ConfigError
from harness.domain.types import (
    BootstrapConfig,
    ExperimentConfig,
    Sample,
    ThresholdConfig,
    TrainingConfig,
)
from harness.ports.artifact_store import ArtifactStorePort
from harness.ports.backbone import BackbonePort
from harness.ports.calibrator import CalibratorPort
from harness.ports.classifier_head import ClassifierHeadPort
from harness.ports.dataset import DatasetPort
from harness.ports.metrics import MetricsPort
from harness.ports.randomness import RandomnessPort
from harness.ports.splitter import SplitterPort
from harness.ports.threshold import ThresholdPort

__all__ = [
    "BackboneChoice",
    "FineTuneBackboneChoice",
    "HeadChoice",
    "RunnerBundle",
    "build_finetune_runner_v1",
    "build_publication_runner_v1",
    "build_v1_runner_sklearn",
    "build_v1_runner_with_fakes",
]

_LABEL_NAMES: tuple[str, ...] = (
    "cardiomegaly",
    "effusion",
    "infiltration",
    "atelectasis",
)
_DATASET_NAME = "fake-cxr-v1"

# Discriminator for ``build_publication_runner_v1(head=...)``.
# - ``"hgbt"`` (default for backwards-compat with the Step 3 PR): the
#   :class:`SklearnGradientBoostingHead`.
# - ``"lr"`` (recommended per the TXRV-embedding ablation): the
#   :class:`SklearnLogisticRegressionHead` with ``class_weight='balanced'``.
#   Lifted macro-F1 from 0.094 (HGBT) to 0.157 on identical TXRV features
#   by rescuing five previously-zero classes; ~zero AUROC cost. The brief
#   defers the default flip to a follow-up PR so the existing baseline
#   numbers remain reproducible without an opt-in.
HeadChoice = Literal["hgbt", "lr"]

# Discriminator for ``build_publication_runner_v1(backbone=...)``.
# - ``"resnet50"`` (default for backwards-compat with the Step 3 PR):
#   :class:`TorchVisionResNet50Backbone`, ImageNet weights, 2048-dim features.
# - ``"txrv-densenet121"``: :class:`TXRVDenseNet121NIHBackbone`, NIH-pretrained
#   DenseNet121, 1024-dim features. CXR-pretrained features lifted macro-AUROC
#   over the ResNet50/ImageNet baseline in the embedding ablation
#   (``/tmp/txrv_embed_ablation/run.py``: 0.643 -> 0.697 on n=4998). Default
#   flip is deferred so existing baselines remain reproducible.
BackboneChoice = Literal["resnet50", "txrv-densenet121"]


@dataclass(frozen=True, slots=True)
class RunnerBundle:
    """Concrete adapter set plus its matching :class:`ExperimentConfig`.

    Returned by both v1 factories. Each field is statically typed against its
    port protocol so callers can unpack ``bundle`` directly into
    :func:`run_experiment` without ``cast`` calls.
    """

    config: ExperimentConfig
    dataset: DatasetPort
    splitter: SplitterPort
    backbone: BackbonePort
    head: ClassifierHeadPort
    calibrator: CalibratorPort
    thresholds: ThresholdPort
    metrics: MetricsPort
    store: ArtifactStorePort
    randomness: RandomnessPort


def _make_fake_dataset(seed: int) -> InMemoryFakeDataset:
    """Build a tiny but multi-label-stratifiable patient-grouped dataset."""
    n_patients = 20
    samples_per_patient = 3
    n_labels = len(_LABEL_NAMES)
    samples: list[Sample] = []
    for p in range(n_patients):
        patient_id = f"P{p:03d}"
        # Deterministic per-patient label vector: each label fires when a
        # SHA-derived bit is set, which gives a reproducible spread of
        # multi-label combinations across the 20 patients.
        digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).digest()
        bits = tuple(int(digest[j] & 1) for j in range(n_labels))
        for s in range(samples_per_patient):
            sample_id = f"{patient_id}_S{s}"
            image_ref = f"img://{seed}/{sample_id}"
            samples.append(
                Sample(
                    sample_id=sample_id,
                    patient_id=patient_id,
                    image_ref=image_ref,
                    labels=bits,
                    metadata={},
                )
            )
    return InMemoryFakeDataset(
        name=_DATASET_NAME, label_names=_LABEL_NAMES, samples=samples
    )


def _make_config(seed: int, *, experiment_name: str, suite: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=experiment_name,
        dataset_name=_DATASET_NAME,
        label_names=_LABEL_NAMES,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=seed,
        bootstrap=BootstrapConfig(n_resamples=8, confidence=0.95, seed=seed + 1),
        threshold=ThresholdConfig(
            method="pr_sweep", shrinkage=0.5, clamp_lo=0.05, clamp_hi=0.95
        ),
        backbone_id="identity-fake",
        head_id="linear-fake",
        calibrator_id="identity-fake",
        artifact_root=f"memory://{suite}",
        notes=f"{suite} runner @ seed={seed}",
    )


def build_v1_runner_with_fakes(seed: int) -> RunnerBundle:
    """Return the full fake adapter set plus a matching ``ExperimentConfig``."""
    dataset = _make_fake_dataset(seed)
    n_labels = len(_LABEL_NAMES)
    return RunnerBundle(
        config=_make_config(seed, experiment_name="v1-fakes", suite="fakes"),
        dataset=dataset,
        splitter=DeterministicFakeSplitter(),
        backbone=IdentityFakeBackbone(image_shape=BYTES_IMAGE_SHAPE),
        head=LinearFakeClassifierHead(n_labels=n_labels),
        calibrator=IdentityFakeCalibrator(),
        thresholds=FixedFakeThreshold(value=0.5),
        metrics=CountingFakeMetrics(),
        store=InMemoryFakeArtifactStore(),
        randomness=SeededRandomness(),
    )


def build_v1_runner_sklearn(seed: int) -> RunnerBundle:
    """Return a runner kit with sklearn-real adapters where applicable.

    Dataset, backbone, and head remain fakes (per the v1 spec, the publication
    pipeline pre-computes features; we exercise the runner's composition
    logic, not real feature extraction). Splitter, calibrator, threshold and
    metrics use the production sklearn adapters.
    """
    dataset = _make_fake_dataset(seed)
    n_labels = len(_LABEL_NAMES)
    return RunnerBundle(
        config=_make_config(seed, experiment_name="v1-sklearn", suite="sklearn"),
        dataset=dataset,
        splitter=IterativeStratifiedPatientSplitter(),
        backbone=IdentityFakeBackbone(image_shape=BYTES_IMAGE_SHAPE),
        head=LinearFakeClassifierHead(n_labels=n_labels),
        calibrator=PerClassPlattCalibrator(),
        thresholds=PrSweepShrinkageThreshold(),
        metrics=BootstrapMetrics(),
        store=InMemoryFakeArtifactStore(),
        randomness=SeededRandomness(),
    )


# ---------------------------------------------------------------------------
# Step 3: publication wiring
# ---------------------------------------------------------------------------


# Default decoded-image size for the publication pipeline. Matches the
# torchvision adapter's internal resize target so the wrapper backbone hands
# the underlying ResNet a tensor it can consume without further interpolation
# (the inner adapter still resizes to 224x224 itself; pre-sizing here keeps
# the wrapper-stack memory footprint predictable).
_PUBLICATION_IMAGE_SIZE: tuple[int, int] = (224, 224)


def _build_inner_backbone(
    *, choice: BackboneChoice, seed: int
) -> tuple[BackbonePort, str, str]:
    """Construct the inner backbone for ``build_publication_runner_v1``.

    Returns ``(backbone, backbone_id, experiment_name_suffix)`` where
    ``backbone_id`` is recorded in the :class:`ExperimentConfig` and
    ``experiment_name_suffix`` differentiates artifact paths between
    variants. The torch / torchxrayvision imports are local to this helper
    so the harness public surface stays importable on machines without the
    ``[experiment]`` extras installed.
    """
    if choice == "resnet50":
        from harness.adapters.torch.backbone import TorchVisionResNet50Backbone

        return (
            TorchVisionResNet50Backbone(seed=seed),
            "torchvision.resnet50",
            "resnet50-imagenet",
        )
    if choice == "txrv-densenet121":
        from harness.adapters.torch.txrv_backbone import TXRVDenseNet121NIHBackbone

        return (
            TXRVDenseNet121NIHBackbone(seed=seed),
            "txrv-densenet121-res224-nih",
            "txrv-densenet121-nih",
        )
    raise ConfigError(
        f"unknown backbone choice {choice!r}; expected one of "
        f"'resnet50', 'txrv-densenet121'"
    )


def build_publication_runner_v1(
    seed: int,
    *,
    nih_csv_path: Path,
    nih_images_dir: Path,
    artifact_root: Path,
    n_samples: int | None = None,
    strict_missing_images: bool = True,
    feature_cache_dir: Path | None = None,
    head: HeadChoice = "hgbt",
    backbone: BackboneChoice = "resnet50",
) -> RunnerBundle:
    """The v1 publication recipe -- see ``PAPER_CHECKLIST.md`` Step 3.

    Wires:

    * :class:`~harness.adapters.fs.nih_dataset.NIHDataset` for the 14-label
      NIH ChestX-ray14 manifest + flat PNG corpus.
    * :class:`~harness.adapters.sklearn.splitter.IterativeStratifiedPatientSplitter`
      for the patient-level multi-label stratified split.
    * One of the supported backbones (selected via the ``backbone`` kwarg):

      - ``"resnet50"`` (default):
        :class:`~harness.adapters.torch.backbone.TorchVisionResNet50Backbone`
        with default ImageNet weights -- 2048-dim features.
      - ``"txrv-densenet121"``:
        :class:`~harness.adapters.torch.txrv_backbone.TXRVDenseNet121NIHBackbone`
        with NIH-pretrained weights -- 1024-dim features. Lifts macro-AUROC
        over the ImageNet baseline in the embedding ablation
        (``/tmp/txrv_embed_ablation/run.py``: 0.643 -> 0.697 on n=4998).

      Both adapters are exposed through a private decoding wrapper that
      bridges the runner's ``BYTES_IMAGE_SHAPE`` pipeline to the backbone's
      NHWC float32 input. When ``feature_cache_dir`` is supplied, the inner
      backbone is wrapped in
      :class:`~harness.adapters.fs.cached_backbone.CachedBackbone` *before*
      the decoding wrapper -- so cache writes happen on the resized
      ``(N, 224, 224, 1)`` tensor that the underlying network actually
      consumes. The cache shards by ``backbone.identifier`` so swapping
      variants does not return stale features.
    * :class:`~harness.adapters.sklearn.head.SklearnGradientBoostingHead` --
      one :class:`HistGradientBoostingClassifier` per output label, seeded
      deterministically from a child seed of ``seed`` (see ``"head"`` label
      below). Replaces the fake linear head that the prior wave wired as a
      placeholder.
    * :class:`~harness.adapters.sklearn.calibrator.PerClassIsotonicCalibrator`
      for per-class isotonic calibration.
    * :class:`~harness.adapters.sklearn.threshold.PrSweepShrinkageThreshold`
      for OOF PR-sweep + shrinkage thresholding.
    * :class:`~harness.adapters.sklearn.metrics.BootstrapMetrics` for macro /
      per-class F1 / AUROC / AUPRC with bootstrap CIs.
    * :class:`~harness.adapters.fs.artifact_store.FilesystemArtifactStore`
      for the four v1 artifacts (model card, thresholds, predictions, metric
      report).
    * :class:`~harness.adapters.fakes.randomness.SeededRandomness` for the
      :class:`RandomnessPort`. The brief calls this ``NumpySeededRandomness``;
      the implemented adapter is named ``SeededRandomness`` and lives in
      ``adapters/fakes/`` -- it is numpy-backed, deterministic, and the only
      adapter implementing the port. A future ``adapters/sklearn/randomness.py``
      could rename it without changing semantics.

    Sub-seeds are derived deterministically via ``RandomnessPort.child_seed``
    with stable labels (``"backbone"``, ``"head"``, ``"bootstrap"``) so two
    invocations with the same ``seed`` produce byte-identical features and
    metrics on the same hardware.

    Args:
        seed: Master seed; sub-seeds are derived per-component.
        nih_csv_path: Absolute path to ``Data_Entry_2017_v2020.csv`` (or the
            16-row synthetic fixture for smoke tests).
        nih_images_dir: Absolute path to the flat directory of NIH PNGs (or
            the fixture's ``images/`` subdirectory).
        artifact_root: Filesystem root under which ``FilesystemArtifactStore``
            writes ``model_card.json``, ``thresholds.json``,
            ``metric_report.json``, and ``predictions/test.csv``.
        n_samples: Optional pilot truncation. ``None`` (default) uses the
            full dataset; a positive int wraps the :class:`NIHDataset` in a
            :class:`_SubsettedDataset` that exposes only the first ``n``
            CSV rows. The CSV is grouped by ``patient_id`` so this is
            patient-leakage-free, but the truncation may end mid-patient
            (patient-block-aligned truncation is deferred to v1.1).
            Values >= ``len(dataset)`` silently fall through to the full
            dataset; values <= 0 are rejected by the caller (the pilot
            CLI), not the factory.
        strict_missing_images: If True (default), :class:`NIHDataset` raises
            :class:`~harness.domain.errors.DataError` when any CSV row
            references a PNG that does not exist on disk. If False, rows
            with missing images are silently dropped (logged at INFO level
            by the underlying adapter). Set False when running on a partial
            NIH dump (e.g. the public 5k sample). The default is True so
            production runs surface dataset integrity issues.
        feature_cache_dir: If set, backbone features are cached to this
            directory (sharded by backbone identifier and image hash). On
            cache hit, feature extraction is skipped -- useful for ablation
            runs where head/calibrator/threshold change but the backbone
            does not. Default ``None`` means no caching (every run
            re-extracts).
        head: Which classifier head to wire. ``"hgbt"`` (default) keeps the
            Step 3 :class:`SklearnGradientBoostingHead` for backwards-compat
            with prior runs. ``"lr"`` selects the
            :class:`SklearnLogisticRegressionHead` (one binary LR per label,
            ``class_weight='balanced'``) which is the **recommended** head
            per the TXRV-embedding ablation: macro-F1 0.094 -> 0.157
            (+0.063) by rescuing five previously-zero classes (Mass,
            Consolidation, Edema, Fibrosis, Pleural_Thickening) at ~zero
            AUROC cost. The default flip is deferred to a follow-up PR so
            existing Step 3 baselines remain reproducible without an opt-in.
        backbone: Which feature-extraction backbone to wire. ``"resnet50"``
            (default) keeps the Step 3 :class:`TorchVisionResNet50Backbone`
            for backwards-compat. ``"txrv-densenet121"`` selects the
            :class:`TXRVDenseNet121NIHBackbone` (NIH-pretrained DenseNet121,
            1024-dim features) -- the **recommended** backbone per the
            embedding ablation: macro-AUROC 0.643 -> 0.697 (+0.054) on
            identical downstream pipeline, no model training.

    Returns:
        A :class:`RunnerBundle` ready to be unpacked into
        :func:`harness.composition.runner.run_experiment`.
    """
    randomness = SeededRandomness()
    backbone_seed = randomness.child_seed(seed, "backbone")
    head_seed = randomness.child_seed(seed, "head")
    bootstrap_seed = randomness.child_seed(seed, "bootstrap")

    nih_config = NIHDatasetConfig(
        csv_path=nih_csv_path,
        images_dir=nih_images_dir,
        image_size=_PUBLICATION_IMAGE_SIZE,
        cache_size=0,  # the decoding wrapper keeps its own cache per-run
        disk_cache_dir=None,
        strict_missing_images=strict_missing_images,
    )
    inner_dataset = NIHDataset(nih_config)
    if n_samples is not None and n_samples > 0:
        # Wrap the NIHDataset in a subset view BEFORE the decoding wrapper.
        # The decoding wrapper reads ``inner.images_dir`` (NIHDataset only)
        # and ``inner.load().samples`` (any DatasetPort) to build its
        # ref->index map; ``_SubsettedDataset`` exposes a truncated
        # ``samples`` tuple while delegating ``images_dir`` via the inner
        # NIHDataset.
        sized_dataset: NIHDataset | _SubsettedDataset = _SubsettedDataset(
            inner_dataset, n_samples
        )
    else:
        sized_dataset = inner_dataset

    cache = DecodedImageCache()
    dataset = _DecodingDataset(
        sized_dataset, cache, image_size=_PUBLICATION_IMAGE_SIZE
    )
    inner_backbone, backbone_id, name_suffix = _build_inner_backbone(
        choice=backbone, seed=backbone_seed
    )
    if feature_cache_dir is not None:
        # Validate the cache path early so configuration mistakes surface as a
        # ``ConfigError`` from the factory, not as an opaque failure mid-run.
        if feature_cache_dir.exists() and not feature_cache_dir.is_dir():
            raise ConfigError(
                f"feature_cache_dir {feature_cache_dir!r} exists but is not a "
                f"directory"
            )
        feature_cache_dir.mkdir(parents=True, exist_ok=True)
        # Order matters: wrap the inner backbone in CachedBackbone *before*
        # the _DecodingBackbone so the cache stores the (224, 224, 1) tensor
        # that the network actually consumes (rather than the runner's
        # (4, 4, 2) bytes-tensor key, which would be useless across runs).
        # CachedBackbone shards by ``inner.identifier`` so resnet50 and
        # txrv-densenet121 features land in distinct subdirectories.
        inner_backbone = CachedBackbone(
            inner=inner_backbone, cache_dir=feature_cache_dir
        )
    decoded_backbone = _DecodingBackbone(
        inner_backbone, cache, image_size=_PUBLICATION_IMAGE_SIZE
    )

    n_labels = len(NIH14_LABELS)
    head_port: ClassifierHeadPort
    if head == "lr":
        head_port = SklearnLogisticRegressionHead(
            n_labels=n_labels, seed=head_seed
        )
        head_id = "sklearn-logistic-regression"
    else:
        head_port = SklearnGradientBoostingHead(
            n_labels=n_labels, seed=head_seed
        )
        head_id = "sklearn-gradient-boosting"
    config = ExperimentConfig(
        experiment_name=f"v1-publication-{name_suffix}",
        dataset_name="nih-cxr14",
        label_names=NIH14_LABELS,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=seed,
        bootstrap=BootstrapConfig(
            n_resamples=8, confidence=0.95, seed=bootstrap_seed
        ),
        threshold=ThresholdConfig(
            method="pr_sweep", shrinkage=0.5, clamp_lo=0.05, clamp_hi=0.95
        ),
        backbone_id=backbone_id,
        head_id=head_id,
        calibrator_id="sklearn-isotonic",
        artifact_root=str(artifact_root),
        notes=(
            f"Publication v1 wiring: NIH-14 + {backbone_id} + "
            f"sklearn {head_id} head + isotonic calibrator + "
            f"PR-sweep thresholds @ seed={seed}. "
            "Calibrator and threshold tuner are co-fit on the same val fold "
            "(see ARCHITECTURE.md §13.1)."
        ),
    )

    return RunnerBundle(
        config=config,
        dataset=dataset,
        splitter=IterativeStratifiedPatientSplitter(),
        backbone=decoded_backbone,
        head=head_port,
        calibrator=PerClassIsotonicCalibrator(),
        thresholds=PrSweepShrinkageThreshold(),
        metrics=BootstrapMetrics(),
        store=FilesystemArtifactStore(root_dir=artifact_root),
        randomness=randomness,
    )


# ---------------------------------------------------------------------------
# v1.1: Fine-tune wiring (FINE_TUNING_DESIGN.md §4.5)
# ---------------------------------------------------------------------------


# Discriminator for ``build_finetune_runner_v1(backbone=...)``. v1.1 ships
# ImageNet-pretrained DenseNet121 and ResNet50 only (FINE_TUNING_DESIGN.md
# §10 answer #6 -- TXRV NIH fine-tuning is deferred to v1.2 to avoid
# fine-tuning on top of leaked weights).
FineTuneBackboneChoice = Literal["densenet121", "resnet50"]


def _make_default_training_config(
    *,
    backbone: FineTuneBackboneChoice,
    n_labels: int,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    augmentations: tuple[str, ...],
    image_size: tuple[int, int],
    checkpoint_dir: Path | None,
) -> TrainingConfig:
    """Default :class:`TrainingConfig` for the v1.1 fine-tune factory.

    Per FINE_TUNING_DESIGN.md §10 answer #5 the default augmentations are
    ``("hflip", "rotate10")`` (the CheXNet recipe). The factory accepts
    a caller-supplied ``augmentations`` tuple to permit ablation override.
    """
    return TrainingConfig(
        backbone_id=backbone,
        n_labels=n_labels,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        optimizer="adamw",
        lr_schedule="cosine",
        warmup_epochs=min(1, n_epochs),
        augmentations=augmentations,
        image_size=image_size,
        checkpoint_dir=checkpoint_dir,
        early_stop_patience=None,
        num_dataloader_workers=0,
    )


def _png_blob_decoder(image_size: tuple[int, int]) -> ImageDecoder:
    """Build a decoder callable mapping ``(image_ref, blob) -> NDArray``.

    Decodes raw PNG bytes via Pillow into a ``(H, W, 1) float32`` array
    in ``[0, 1]`` -- the same shape :class:`NIHImageLoader.decode` produces
    for the frozen-feature publication path. The ``image_ref`` is used
    purely as a cache key inside :class:`_InMemoryTrainingDataset`; this
    decoder re-decodes the supplied bytes per call.
    """
    h, w = image_size

    def _decoder(_ref: str, blob: bytes) -> NDArray[np.float32]:
        with Image.open(BytesIO(blob)) as img:
            grayscale = img.convert("L")
            resized = grayscale.resize((w, h), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
        result: NDArray[np.float32] = np.clip(arr, 0.0, 1.0).reshape(h, w, 1)
        return result

    return _decoder


def build_finetune_runner_v1(
    seed: int,
    *,
    nih_csv_path: Path,
    nih_images_dir: Path,
    artifact_root: Path,
    n_samples: int | None = None,
    strict_missing_images: bool = True,
    backbone: FineTuneBackboneChoice = "densenet121",
    n_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    augmentations: tuple[str, ...] = ("hflip", "rotate10"),
    image_size: tuple[int, int] = (224, 224),
    checkpoint_dir: Path | None = None,
) -> FineTuneRunnerBundle:
    """v1.1 fine-tune recipe (FINE_TUNING_DESIGN.md §4.5).

    Parallel to :func:`build_publication_runner_v1` but wires the new
    :class:`TorchFineTuneTrainer` adapter in place of backbone + head.
    The trainer subsumes feature extraction and head fitting; downstream
    calibrator/threshold/metrics chain is byte-identical to the
    frozen-feature path.

    Args:
        seed: Master seed; sub-seeds for trainer and split are derived.
        nih_csv_path: Absolute path to ``Data_Entry_2017_v2020.csv``.
        nih_images_dir: Absolute path to the flat directory of NIH PNGs.
        artifact_root: Filesystem root for the four v1 artifacts.
        n_samples: Optional pilot truncation (same semantics as
            :func:`build_publication_runner_v1`). When set, the in-memory
            training dataset's 20000-row cap (FINE_TUNING_DESIGN.md §10
            answer #3) means ``n_samples * (1 - val - test)`` must be
            <= 20000; the factory does NOT enforce this -- it surfaces
            via :class:`ConfigError` from
            :class:`_InMemoryTrainingDataset` at runtime.
        strict_missing_images: If True (default), raises
            :class:`DataError` when a CSV row references a missing PNG.
        backbone: ``"densenet121"`` (default) or ``"resnet50"``.
        n_epochs: Default 1 -- matches the smoke gate per
            FINE_TUNING_DESIGN.md §10 answer #1.
        batch_size: Default 32 -- comfortable for MPS at 224x224.
        learning_rate: Default 1e-4 -- standard AdamW fine-tune LR for
            ImageNet-pretrained CNNs.
        augmentations: Default ``("hflip", "rotate10")`` (CheXNet recipe).
        image_size: Default ``(224, 224)`` -- matches the configured
            backbones' expected input size.
        checkpoint_dir: Optional checkpoint directory. ``None`` disables
            checkpointing (training restarts every call).

    Returns:
        A :class:`FineTuneRunnerBundle` ready to be unpacked into
        :func:`harness.composition.runner.run_finetune_experiment`.
    """
    # Local import: TorchFineTuneTrainer pulls torch + torchvision at import
    # time; keeping the import inside the factory means the harness public
    # surface stays usable on machines without the [experiment] extras.
    from harness.adapters.torch.trainer import TorchFineTuneTrainer

    randomness = SeededRandomness()
    bootstrap_seed = randomness.child_seed(seed, "bootstrap")
    nih_config = NIHDatasetConfig(
        csv_path=nih_csv_path,
        images_dir=nih_images_dir,
        image_size=image_size,
        cache_size=0,
        disk_cache_dir=None,
        strict_missing_images=strict_missing_images,
    )
    inner_dataset = NIHDataset(nih_config)
    sized_dataset: NIHDataset | _SubsettedDataset
    if n_samples is not None and n_samples > 0:
        sized_dataset = _SubsettedDataset(inner_dataset, n_samples)
    else:
        sized_dataset = inner_dataset
    n_labels = len(NIH14_LABELS)
    training_cfg = _make_default_training_config(
        backbone=backbone,
        n_labels=n_labels,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        augmentations=augmentations,
        image_size=image_size,
        checkpoint_dir=checkpoint_dir,
    )
    config = ExperimentConfig(
        experiment_name=f"v1.1-finetune-{backbone}",
        dataset_name="nih-cxr14",
        label_names=NIH14_LABELS,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=seed,
        bootstrap=BootstrapConfig(
            n_resamples=8, confidence=0.95, seed=bootstrap_seed
        ),
        threshold=ThresholdConfig(
            method="pr_sweep", shrinkage=0.5, clamp_lo=0.05, clamp_hi=0.95
        ),
        backbone_id=f"torch.finetune.{backbone}.v1",
        head_id="torch.finetune.linear.v1",
        calibrator_id="sklearn-isotonic",
        artifact_root=str(artifact_root),
        notes=(
            f"Fine-tune v1.1: NIH-14 + {backbone} (ImageNet init) + "
            f"AdamW + cosine LR + augmentations={augmentations} @ seed={seed}. "
            "Calibrator and threshold tuner are co-fit on the same val fold "
            "(see ARCHITECTURE.md §13.1)."
        ),
        training=training_cfg,
    )
    decoder = _png_blob_decoder(image_size)
    return FineTuneRunnerBundle(
        config=config,
        dataset=sized_dataset,
        splitter=IterativeStratifiedPatientSplitter(),
        trainer=TorchFineTuneTrainer(),
        calibrator=PerClassIsotonicCalibrator(),
        thresholds=PrSweepShrinkageThreshold(),
        metrics=BootstrapMetrics(),
        store=FilesystemArtifactStore(root_dir=artifact_root),
        randomness=randomness,
        decoder=decoder,
    )
