"""Trainer port: end-to-end fine-tuning surface for v1.1.

Per ``harness/docs/FINE_TUNING_DESIGN.md``, fine-tuning lives behind a new
port rather than extending :class:`~harness.ports.backbone.BackbonePort`.
Three Protocols ship here:

* :class:`TrainingDatasetPort` -- a finite-length iterable of decoded
  ``(image, labels)`` rows. Built by the composition root from a
  :class:`~harness.ports.dataset.DatasetPort` plus a split's index list;
  the trainer never sees raw image bytes or :class:`~harness.domain.types.Sample`
  records.
* :class:`TrainerPort` -- consumes a training dataset, a validation
  dataset, a :class:`TrainingConfig` (added to ``harness/domain/types.py``
  in the implementation PR), and a seed; returns a
  :class:`TrainedClassifierPort` plus a :class:`TrainingResult`
  bookkeeping record.
* :class:`TrainedClassifierPort` -- the eval-ready model returned by the
  trainer. Exposes only ``predict_proba`` so the runner can route its
  output into the existing calibrator/threshold/metrics chain unchanged.

This file is the **stub** that ships in the design PR. Concrete adapters
land in a follow-up implementation PR. The
:class:`TrainingConfig` and :class:`TrainingResult` domain types referenced
in the docstrings are added to ``harness/domain/types.py`` in the same
implementation PR; until then this module imports them under
``TYPE_CHECKING`` only so the stub is importable in the v1 codebase
without forward-ref breakage.

Determinism contract (per FINE_TUNING_DESIGN.md §7):

* :meth:`TrainerPort.fit` must be a pure function of
  ``(training_dataset, validation_dataset, config, seed)``. Same inputs
  produce byte-identical model weights.
* No global RNG mutation: adapters use scoped ``torch.Generator`` /
  ``numpy.random.Generator`` instances threaded from ``seed``.
* MPS runs are intra-device reproducible only; the model card's notes
  field must record the device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # ``TrainingConfig`` / ``TrainingResult`` are added to ``harness.domain.types``
    # in the v1.1 implementation PR. The design PR ships only this stub; no
    # adapter exists yet that would import the symbols at runtime. Keeping the
    # imports under ``TYPE_CHECKING`` lets the design PR land without
    # forward-introducing the domain types ahead of the implementation that
    # will validate them (TDD: domain validators land first in the impl PR).
    from harness.domain.types import TrainingConfig, TrainingResult

__all__ = [
    "TrainedClassifierPort",
    "TrainerPort",
    "TrainingDatasetPort",
]


@runtime_checkable
class TrainingDatasetPort(Protocol):
    """A finite-length iterable of decoded ``(image, labels)`` rows.

    Construction is the composition root's responsibility: the runner wraps
    an existing :class:`~harness.ports.dataset.DatasetPort` plus a split's
    index list into a thin in-memory adapter (``_InMemoryTrainingDataset``,
    landing in the implementation PR). The trainer reads images and labels
    via ``__getitem__`` and ``__len__`` (PyTorch's ``DataLoader``-friendly
    surface) without ever seeing raw bytes or :class:`~harness.domain.types.Sample`
    records.

    ``__getitem__`` is intentionally the only access pattern; no
    ``__iter__`` (PyTorch derives one automatically), no metadata accessors
    (the trainer doesn't need them), no patient ID surface (splitting
    already happened upstream).
    """

    def __len__(self) -> int:
        """Number of rows in this view.

        Adapters must return a non-negative integer. ``0`` is permitted at
        the port level but :class:`TrainerPort.fit` rejects empty
        ``training_dataset`` values with :class:`AdapterError`.
        """
        ...

    def __getitem__(
        self, index: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
        """Return ``(image, labels)`` for row ``index``.

        ``image`` is shape ``(H, W, C)`` ``float32`` in ``[0, 1]``;
        ``labels`` is shape ``(K,)`` ``int8`` multi-hot (each entry is
        ``0`` or ``1``). All rows in a single dataset must agree on
        ``(H, W, C)`` and ``K`` -- callers may rely on shape consistency.

        Out-of-range indices raise :class:`IndexError` (mirrors Python's
        sequence contract; PyTorch's ``DataLoader`` depends on it).
        """
        ...


@runtime_checkable
class TrainedClassifierPort(Protocol):
    """A model that has been trained end-to-end and is ready for inference.

    Returned by :meth:`TrainerPort.fit`. The runner treats this as the
    backbone-plus-head replacement: instead of calling
    :meth:`~harness.ports.backbone.BackbonePort.extract` then
    :meth:`~harness.ports.classifier_head.ClassifierHeadPort.predict_proba`,
    it calls :meth:`predict_proba` directly on raw NHWC images.

    Implementations must be eval-only and pure functions of their inputs.
    Two consecutive calls with identical inputs return byte-identical
    outputs (no dropout, no batchnorm drift).
    """

    @property
    def n_labels(self) -> int:
        """Number of output classes (must equal ``len(label_names)``)."""
        ...

    @property
    def identifier(self) -> str:
        """Stable string identifier (e.g. ``"torch.finetune.densenet121.v1"``).

        Recorded in :class:`~harness.domain.types.ModelCard` as the
        backbone identifier so downstream lineage tracking sees the
        trained model's recipe.
        """
        ...

    def predict_proba(
        self, images: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Map ``(N, H, W, C)`` images to ``(N, n_labels)`` probabilities.

        All output values must lie in ``[0, 1]``. Adapters raise
        :class:`~harness.domain.errors.AdapterError` on shape/dtype
        mismatch (e.g. ``ndim != 4``, channel count not in ``{1, 3}``).
        Empty batches (``images.shape[0] == 0``) return an empty
        ``(0, n_labels)`` array.
        """
        ...


@runtime_checkable
class TrainerPort(Protocol):
    """End-to-end model trainer.

    :meth:`fit` consumes a training dataset, a validation dataset, a
    :class:`~harness.domain.types.TrainingConfig`, and a seed; returns a
    :class:`TrainedClassifierPort` plus a
    :class:`~harness.domain.types.TrainingResult` bookkeeping record. The
    trainer owns:

    * The ``nn.Module`` (resolved from ``config.backbone_id``).
    * The optimizer (resolved from ``config.optimizer``).
    * The LR schedule (``config.lr_schedule`` + ``config.warmup_epochs``).
    * The ``torch.utils.data.DataLoader`` construction.
    * The augmentation pipeline (``config.augmentations``).
    * The loss function (``nn.BCEWithLogitsLoss`` in v1.1; the
      :class:`TrainingConfig` does not expose a ``loss=`` parameter --
      see FINE_TUNING_DESIGN.md §8).
    * The checkpoint format (``config.checkpoint_dir`` if set).

    The trainer does NOT own dataset construction or train/val splitting --
    those flow in via the ``training_dataset`` and ``validation_dataset``
    keyword arguments.
    """

    @property
    def identifier(self) -> str:
        """Stable identifier for the training recipe.

        Examples: ``"torch.finetune.densenet121.v1"``,
        ``"torch.finetune.resnet50.v1"``. Recorded in the model card so a
        future reader can reproduce the run from the recipe name plus the
        :class:`TrainingConfig`.
        """
        ...

    def fit(
        self,
        *,
        training_dataset: TrainingDatasetPort,
        validation_dataset: TrainingDatasetPort,
        config: TrainingConfig,
        seed: int,
    ) -> tuple[TrainedClassifierPort, TrainingResult]:
        """Train the model and return the eval-ready classifier + bookkeeping.

        Implementations must:

        * Be deterministic given ``(training_dataset, validation_dataset,
          config, seed)``. Same inputs -> byte-identical model weights
          (CPU/CUDA only; MPS runs are intra-device reproducible -- see
          FINE_TUNING_DESIGN.md §7).
        * NOT touch global RNG state outside what is reachable via
          ``seed``. No ``torch.manual_seed`` in module scope, no
          ``np.random.seed``. Use scoped ``torch.Generator`` / numpy
          ``Generator`` instances.
        * Wrap third-party exceptions in
          :class:`~harness.domain.errors.AdapterError`.
        * Respect ``config.checkpoint_dir`` for resume-from-checkpoint:
          on a second invocation with the same dir, training resumes from
          the latest checkpoint rather than restarting. Mismatched
          ``config_hash`` between an existing checkpoint and the supplied
          ``config`` must raise :class:`AdapterError`; the operator is
          expected to use a different ``checkpoint_dir`` for a logically
          different run.
        * Reject empty training datasets (``len(training_dataset) == 0``)
          with :class:`AdapterError`.
        * Reject label-shape mismatches (``training_dataset[0][1].shape[0]
          != config.n_labels``) with :class:`AdapterError`.
        """
        ...
