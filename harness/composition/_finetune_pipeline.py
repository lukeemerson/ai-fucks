"""Composition-internal helpers for the v1.1 fine-tune pipeline.

Per FINE_TUNING_DESIGN.md §4.3 the fine-tune runner builds a
:class:`TrainingDatasetPort` from an existing :class:`DatasetPort` plus a
split's index list. v1.1 ships an in-memory variant only; a streaming
variant is deferred to v1.2.

The 20000-row cap (FINE_TUNING_DESIGN.md §10 answer #3) is enforced at
construction time so misconfigured runs fail fast with :class:`ConfigError`
rather than OOMing mid-training.

This module is the only layer that may bridge the dataset port (for the
byte source) and an image-decoding callable supplied by the composition
root. The decoder is a callable rather than an adapter import so the
trainer pipeline stays decoupled from any specific filesystem image
loader (the composition root wires the NIH PIL decoder; tests inject a
trivial deterministic decoder).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from harness.domain.errors import AdapterError, ConfigError
from harness.domain.types import ExperimentConfig
from harness.ports.artifact_store import ArtifactStorePort
from harness.ports.calibrator import CalibratorPort
from harness.ports.dataset import DatasetPort
from harness.ports.metrics import MetricsPort
from harness.ports.randomness import RandomnessPort
from harness.ports.splitter import SplitterPort
from harness.ports.threshold import ThresholdPort
from harness.ports.trainer import TrainerPort

__all__ = [
    "FineTuneRunnerBundle",
    "ImageDecoder",
    "_InMemoryTrainingDataset",
    "_MAX_IN_MEMORY_ROWS",
]


# Hard upper bound on the row count for the in-memory training dataset.
# Per FINE_TUNING_DESIGN.md §10 answer #3 (the approved §10 sign-off):
# v1.1 ships in-memory only; > 20000 rows must fail fast at construction
# time with ConfigError so a misconfigured ablation does not silently OOM.
_MAX_IN_MEMORY_ROWS: int = 20_000


# An image decoder maps ``(image_ref, image_bytes) -> (H, W, C) float32 [0, 1]``.
# Supplied by the composition root: tests use a trivial bytes-to-array
# decoder; production wires the NIH PIL loader.
ImageDecoder = Callable[[str, bytes], NDArray[np.float32]]


class _InMemoryTrainingDataset:
    """``TrainingDatasetPort`` backed by an in-memory list of byte refs.

    Construction takes the underlying :class:`DatasetPort`, the split's
    index list, the configured ``n_labels`` and ``image_size``, and an
    image-decoder callable. ``__getitem__`` decodes lazily on first call
    and caches the decoded ``(H, W, C) float32`` row so subsequent epochs
    pay zero decode cost.

    The 20000-row cap is enforced at construction. Streaming variants
    are deferred to v1.2.
    """

    __slots__ = (
        "_decoder",
        "_image_size",
        "_indices",
        "_n_labels",
        "_rows_cache",
        "_samples",
        "_source",
    )

    def __init__(
        self,
        *,
        source: DatasetPort,
        indices: Sequence[int],
        n_labels: int,
        image_size: tuple[int, int],
        decoder: ImageDecoder,
    ) -> None:
        if len(indices) > _MAX_IN_MEMORY_ROWS:
            raise ConfigError(
                f"_InMemoryTrainingDataset: row count {len(indices)} exceeds "
                f"the v1.1 cap of {_MAX_IN_MEMORY_ROWS} (FINE_TUNING_DESIGN.md "
                "§10 answer #3). Streaming variant is deferred to v1.2; for "
                "larger slices, lower the slice size or wait for v1.2."
            )
        self._source: DatasetPort = source
        self._indices: tuple[int, ...] = tuple(indices)
        self._n_labels: int = n_labels
        self._image_size: tuple[int, int] = image_size
        self._decoder: ImageDecoder = decoder
        # Snapshot the underlying samples once so we don't pay the
        # ``DatasetPort.load`` cost per row.
        ds = source.load()
        self._samples = ds.samples
        # Lazy cache for decoded ``(image, labels)`` pairs. The training
        # loop iterates many epochs over the same indices; caching saves
        # repeated PNG decode + resize work without exceeding the 20k cap
        # on row count (each entry is one decoded image).
        self._rows_cache: dict[
            int, tuple[NDArray[np.float32], NDArray[np.int8]]
        ] = {}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(
        self, index: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
        if index < 0 or index >= len(self._indices):
            raise IndexError(
                f"index {index} out of range [0, {len(self._indices)})"
            )
        cached = self._rows_cache.get(index)
        if cached is not None:
            return cached
        underlying_index = self._indices[index]
        sample = self._samples[underlying_index]
        if len(sample.labels) != self._n_labels:
            raise AdapterError(
                f"_InMemoryTrainingDataset row {index}: sample {sample.sample_id!r} "
                f"has {len(sample.labels)} labels but config.n_labels="
                f"{self._n_labels}"
            )
        blob = self._source.get_image_bytes(sample.image_ref)
        image = self._decoder(sample.image_ref, blob)
        if image.dtype != np.float32:
            image = image.astype(np.float32, copy=False)
        labels = np.asarray(sample.labels, dtype=np.int8)
        row = (image, labels)
        self._rows_cache[index] = row
        return row


@dataclass(frozen=True, slots=True)
class FineTuneRunnerBundle:
    """Concrete adapter set for ``run_finetune_experiment``.

    Mirrors :class:`harness.composition.factories.RunnerBundle` but drops
    the ``backbone`` and ``head`` fields (the trainer subsumes both) and
    adds a ``trainer`` field. Constructed by
    :func:`harness.composition.factories.build_finetune_runner_v1`.

    Per FINE_TUNING_DESIGN.md §4.5.
    """

    config: ExperimentConfig
    dataset: DatasetPort
    splitter: SplitterPort
    trainer: TrainerPort
    calibrator: CalibratorPort
    thresholds: ThresholdPort
    metrics: MetricsPort
    store: ArtifactStorePort
    randomness: RandomnessPort
    decoder: ImageDecoder
