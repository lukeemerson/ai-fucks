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

from harness.adapters.fakes.artifact_store import InMemoryFakeArtifactStore
from harness.adapters.fakes.backbone import IdentityFakeBackbone
from harness.adapters.fakes.calibrator import IdentityFakeCalibrator
from harness.adapters.fakes.classifier_head import LinearFakeClassifierHead
from harness.adapters.fakes.dataset import InMemoryFakeDataset
from harness.adapters.fakes.metrics import CountingFakeMetrics
from harness.adapters.fakes.randomness import SeededRandomness
from harness.adapters.fakes.splitter import DeterministicFakeSplitter
from harness.adapters.fakes.threshold import FixedFakeThreshold
from harness.adapters.sklearn.calibrator import PerClassPlattCalibrator
from harness.adapters.sklearn.metrics import BootstrapMetrics
from harness.adapters.sklearn.splitter import IterativeStratifiedPatientSplitter
from harness.adapters.sklearn.threshold import PrSweepShrinkageThreshold
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
