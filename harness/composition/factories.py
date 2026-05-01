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
from pathlib import Path

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
from harness.adapters.fs.nih_csv import NIH14_LABELS
from harness.adapters.fs.nih_dataset import NIHDataset, NIHDatasetConfig
from harness.adapters.sklearn.calibrator import (
    PerClassIsotonicCalibrator,
    PerClassPlattCalibrator,
)
from harness.adapters.sklearn.head import SklearnGradientBoostingHead
from harness.adapters.sklearn.metrics import BootstrapMetrics
from harness.adapters.sklearn.splitter import IterativeStratifiedPatientSplitter
from harness.adapters.sklearn.threshold import PrSweepShrinkageThreshold
from harness.composition._publication_pipeline import (
    DecodedImageCache,
    _DecodingBackbone,
    _DecodingDataset,
    _SubsettedDataset,
)
from harness.composition.runner import BYTES_IMAGE_SHAPE
from harness.domain.types import (
    BootstrapConfig,
    ExperimentConfig,
    Sample,
    ThresholdConfig,
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
    "RunnerBundle",
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


def build_publication_runner_v1(
    seed: int,
    *,
    nih_csv_path: Path,
    nih_images_dir: Path,
    artifact_root: Path,
    n_samples: int | None = None,
) -> RunnerBundle:
    """The v1 publication recipe -- see ``PAPER_CHECKLIST.md`` Step 3.

    Wires:

    * :class:`~harness.adapters.fs.nih_dataset.NIHDataset` for the 14-label
      NIH ChestX-ray14 manifest + flat PNG corpus.
    * :class:`~harness.adapters.sklearn.splitter.IterativeStratifiedPatientSplitter`
      for the patient-level multi-label stratified split.
    * :class:`~harness.adapters.torch.backbone.TorchVisionResNet50Backbone`
      with default ImageNet weights and auto-selected device, exposed through
      a private decoding wrapper that bridges the runner's
      ``BYTES_IMAGE_SHAPE`` pipeline to the backbone's NHWC float32 input.
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

    The :class:`TorchVisionResNet50Backbone` import is local to this function
    so that the rest of the harness public surface remains importable on
    machines without the ``[experiment]`` extras installed.

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

    Returns:
        A :class:`RunnerBundle` ready to be unpacked into
        :func:`harness.composition.runner.run_experiment`.
    """
    # Local import: torchvision is gated behind the ``[experiment]`` extras
    # group. Importing inside the function keeps ``harness`` importable on
    # CI / dev machines that haven't installed torch.
    from harness.adapters.torch.backbone import TorchVisionResNet50Backbone

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
        strict_missing_images=True,
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
    inner_backbone = TorchVisionResNet50Backbone(seed=backbone_seed)
    backbone = _DecodingBackbone(
        inner_backbone, cache, image_size=_PUBLICATION_IMAGE_SIZE
    )

    n_labels = len(NIH14_LABELS)
    config = ExperimentConfig(
        experiment_name="v1-publication-resnet50-imagenet",
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
        backbone_id="torchvision.resnet50",
        head_id="sklearn-gradient-boosting",
        calibrator_id="sklearn-isotonic",
        artifact_root=str(artifact_root),
        notes=(
            "Publication v1 wiring: NIH-14 + ResNet50 ImageNet + "
            "sklearn HistGradientBoosting head + isotonic calibrator + "
            f"PR-sweep thresholds @ seed={seed}. "
            "Calibrator and threshold tuner are co-fit on the same val fold "
            "(see ARCHITECTURE.md §13.1)."
        ),
    )

    return RunnerBundle(
        config=config,
        dataset=dataset,
        splitter=IterativeStratifiedPatientSplitter(),
        backbone=backbone,
        head=SklearnGradientBoostingHead(n_labels=n_labels, seed=head_seed),
        calibrator=PerClassIsotonicCalibrator(),
        thresholds=PrSweepShrinkageThreshold(),
        metrics=BootstrapMetrics(),
        store=FilesystemArtifactStore(root_dir=artifact_root),
        randomness=randomness,
    )
