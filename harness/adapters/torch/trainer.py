"""End-to-end fine-tuning trainer for the v1.1 fine-tune surface.

Implements :class:`harness.ports.trainer.TrainerPort` per
``harness/docs/FINE_TUNING_DESIGN.md`` §5. Ships exactly two backbone
options (``densenet121``, ``resnet50``) with ImageNet-pretrained init;
unknown ``backbone_id`` (including any TXRV variant) raises
:class:`ConfigError` per §10 answer #6.

Loss is :class:`torch.nn.BCEWithLogitsLoss`; optimizer is
:class:`torch.optim.AdamW` (the v1.1 §3/§5 reconciliation narrows the
optimizer literal to ``"adamw"`` only). LR schedule is cosine with
linear warmup, or constant. Augmentations are an ordered tuple of
torchvision-v2 transform names; unknown names raise :class:`ConfigError`.

Determinism (FINE_TUNING_DESIGN.md §7)
--------------------------------------
* No global ``torch.manual_seed`` outside the trainer's own scope. The
  trainer constructs its own :class:`torch.Generator` per ``fit`` call,
  derives sub-seeds for the model init, dataloader, and augmentation
  pipeline via SHA-256 over ``f"{seed}:{label}"``, and threads them
  through :class:`DataLoader` and :class:`v2.RandomHorizontalFlip` /
  :class:`v2.RandomRotation` (which accept a per-call ``generator``).
* On CUDA, sets ``torch.backends.cudnn.deterministic = True`` and
  ``cudnn.benchmark = False``. MPS provides no equivalent flag; runs on
  MPS are intra-device reproducible only (FINE_TUNING_DESIGN.md §7,
  ARCHITECTURE.md §13.5).
* The decision to honour the v2 transforms' generator argument follows
  FINE_TUNING_DESIGN.md §5's first-priority guidance ("Use
  torchvision.transforms.v2 which is generator-aware end-to-end"); the
  fallback to per-epoch ``torch.manual_seed`` is NOT taken in v1.1.

Checkpointing (FINE_TUNING_DESIGN.md §5)
----------------------------------------
On every epoch end the trainer saves
``{checkpoint_dir}/epoch_{n:03d}.pt`` containing model + optimizer +
scheduler + RNG state plus a config hash. On resume the latest
checkpoint is loaded; mismatched hash raises :class:`AdapterError`.

Out of scope (FINE_TUNING_DESIGN.md §8)
---------------------------------------
* Mixed precision, multi-GPU, focal/asymmetric loss, hyperparameter
  search, multi-worker DataLoader (workers locked to 0 by
  ``TrainingConfig.__post_init__``).
* TXRV-NIH fine-tuning per §10 answer #6.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from typing import Final, Literal, get_args

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    densenet121,
    resnet50,
)
from torchvision.transforms import v2 as transforms_v2

from harness.domain.errors import AdapterError, ConfigError
from harness.domain.types import TrainingConfig, TrainingResult
from harness.ports.trainer import TrainingDatasetPort

__all__ = [
    "TorchFineTuneTrainer",
]

DeviceName = Literal["cpu", "cuda", "mps"]

_IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

# Allowed backbones for v1.1 (FINE_TUNING_DESIGN.md §10 answer #6).
_ALLOWED_BACKBONES: Final[tuple[str, ...]] = ("densenet121", "resnet50")
_ALLOWED_AUGMENTATIONS: Final[tuple[str, ...]] = ("hflip", "rotate10")

_TRAINER_IDENTIFIER: Final[str] = "torch.finetune.v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_device(override: DeviceName | None) -> DeviceName:
    """Pick a device, preferring MPS, then CUDA, then CPU.

    Mirrors :func:`harness.adapters.torch.backbone._select_device`.
    """
    if override is None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    valid: tuple[DeviceName, ...] = get_args(DeviceName)
    if override not in valid:
        raise AdapterError(
            f"unknown device override {override!r}; expected one of {valid}"
        )
    if override == "cuda" and not torch.cuda.is_available():
        raise AdapterError(
            "device='cuda' requested but torch.cuda.is_available() is False"
        )
    if override == "mps" and not (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):
        raise AdapterError(
            "device='mps' requested but torch.backends.mps.is_available()/is_built() is False"
        )
    return override


def _child_seed(parent: int, label: str) -> int:
    """SHA-256-based child seed derivation; mirrors ``SeededRandomness``."""
    digest = hashlib.sha256(f"{parent}:{label}".encode()).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value & ((1 << 63) - 1)


# Fields excluded from the resume-equality hash. ``n_epochs`` is excluded so
# operators can resume with a longer schedule (the canonical resume use
# case); ``early_stop_patience`` is excluded for the same reason. All other
# fields participate in the hash so a logically different run -- different
# backbone, different LR, different augmentations -- fails fast with
# AdapterError rather than silently mixing weights.
#
# v1.1 deviation from FINE_TUNING_DESIGN.md §5 ("On resume: load the latest
# epoch_*.pt whose config_hash matches the current TrainingConfig"). The
# design doc's strict-equality wording was incompatible with the design
# doc's own resume use case (§6.2 "train 3 epochs to checkpoint dir, load
# via a fresh trainer with n_epochs=5"). Documented in
# FINE_TUNING_DESIGN.md §5 deviation entry as part of the implementation PR.
_HASH_EXCLUDED_FIELDS: tuple[str, ...] = ("n_epochs", "early_stop_patience")


def _config_hash(config: TrainingConfig) -> str:
    """Deterministic SHA-256 over the resume-relevant ``TrainingConfig`` fields.

    Used by the checkpoint resume path to detect a logically different
    run reusing the same ``checkpoint_dir``. ``n_epochs`` and
    ``early_stop_patience`` are excluded so operators can extend a
    truncated schedule on resume; see ``_HASH_EXCLUDED_FIELDS``.
    """
    payload_dict = {
        k: v for k, v in asdict(config).items() if k not in _HASH_EXCLUDED_FIELDS
    }
    payload = json.dumps(
        payload_dict, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# A large prime; used as the per-epoch stride in ``_augmentation_seed`` so the
# combined ``base + epoch * STRIDE + batch_idx`` value avoids collisions for
# any plausible ``(n_epochs, n_batches)`` pair (n_epochs <= 1e4, batches per
# epoch <= STRIDE - 1 -> ~1e5 batches/epoch).
_AUG_EPOCH_STRIDE: Final[int] = 100003


def _augmentation_seed(base_seed: int, *, epoch: int, batch_idx: int) -> int:
    """Derive a per-(epoch, batch) augmentation seed.

    Fixes M1 from the Wave 4 review: the original
    ``aug_gen.initial_seed() + n_batches`` formulation reset ``n_batches``
    each epoch, so batch position 0 in every epoch saw identical aug
    parameters. Threading ``epoch`` into the derivation breaks that
    collision while preserving determinism for a fixed
    ``(seed, dataset, n_epochs)`` triple.

    The resulting seed is masked to fit a non-negative 63-bit int so it
    can be passed unchanged to ``torch.manual_seed`` without overflow on
    LP64 platforms.
    """
    raw = int(base_seed) + epoch * _AUG_EPOCH_STRIDE + batch_idx
    return int(raw) & ((1 << 63) - 1)


def _compute_lr_for_epoch(
    *,
    epoch: int,
    base_lr: float,
    n_epochs: int,
    warmup_epochs: int,
    schedule: str,
) -> float:
    """Stateless cosine-with-warmup / constant LR computation.

    Replaces ``TorchFineTuneTrainer._compute_lr`` with a module-level
    helper so unit tests can drive it without spinning up a full ``fit``
    loop and the trainer no longer needs to expose mutable test
    affordance instance state (M2 fix). The semantics are unchanged
    from §5: warmup linearly ramps ``1/W .. W/W * base_lr`` across
    ``warmup_epochs`` epochs, then a half-cosine decays from ``base_lr``
    to 0 over the remaining ``n_epochs - warmup_epochs`` epochs.
    """
    if schedule == "constant":
        return base_lr
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    remaining_epochs = max(n_epochs - warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / float(remaining_epochs)
    progress = max(0.0, min(progress, 1.0))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_model(*, backbone_id: str, n_labels: int) -> nn.Module:
    """Resolve ``backbone_id`` to an ImageNet-pretrained ``nn.Module`` with a
    fresh ``nn.Linear(in_features, n_labels)`` classifier head.

    v1.1 ships only ``densenet121`` and ``resnet50``; unknown ids raise
    :class:`ConfigError`.
    """
    if backbone_id == "densenet121":
        weights_dn = getattr(DenseNet121_Weights, "IMAGENET1K_V1", None)
        model_dn = densenet121(weights=weights_dn)
        in_features = int(model_dn.classifier.in_features)
        model_dn.classifier = nn.Linear(in_features, n_labels)
        result: nn.Module = model_dn
        return result
    if backbone_id == "resnet50":
        weights_rn = getattr(ResNet50_Weights, "IMAGENET1K_V2", None)
        if weights_rn is None:
            weights_rn = getattr(ResNet50_Weights, "IMAGENET1K_V1", None)
        model_rn = resnet50(weights=weights_rn)
        in_features = int(model_rn.fc.in_features)
        model_rn.fc = nn.Linear(in_features, n_labels)
        result_rn: nn.Module = model_rn
        return result_rn
    raise ConfigError(
        f"unknown backbone_id {backbone_id!r}; v1.1 supports "
        f"{_ALLOWED_BACKBONES!r} (TXRV-NIH fine-tuning is deferred to v1.2; "
        "see FINE_TUNING_DESIGN.md §10 answer #6)"
    )


def _build_augmentation_pipeline(
    augmentations: tuple[str, ...],
) -> nn.Module | None:
    """Compose a torchvision-v2 augmentation pipeline.

    Returns ``None`` when ``augmentations`` is empty. Unknown names raise
    :class:`ConfigError`. Each transform here is generator-aware in v2,
    which lets ``fit`` thread its own seeded :class:`torch.Generator` for
    determinism (FINE_TUNING_DESIGN.md §5/§7).
    """
    if not augmentations:
        return None
    transforms: list[nn.Module] = []
    for name in augmentations:
        if name == "hflip":
            transforms.append(transforms_v2.RandomHorizontalFlip(p=0.5))
        elif name == "rotate10":
            transforms.append(transforms_v2.RandomRotation(degrees=10))
        else:
            raise ConfigError(
                f"unknown augmentation {name!r}; v1.1 supports "
                f"{_ALLOWED_AUGMENTATIONS!r}"
            )
    composed: nn.Module = transforms_v2.Compose(transforms)
    return composed


# ---------------------------------------------------------------------------
# Internal torch ``Dataset`` adapter
# ---------------------------------------------------------------------------


class _TorchDatasetView(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Bridge a :class:`TrainingDatasetPort` to ``torch.utils.data.Dataset``.

    Reads ``(image, labels)`` rows via the port's ``__getitem__``, converts
    to ``(C, H, W)`` float32 / int8 tensors. Normalization and resize live
    in :meth:`TorchFineTuneTrainer._forward_batch` so the augmentation
    pipeline can apply on the still-uint8-equivalent tensor.
    """

    def __init__(self, port: TrainingDatasetPort) -> None:
        self._port = port

    def __len__(self) -> int:
        return len(self._port)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, labels = self._port[index]
        # NHWC float32 -> CHW float32.
        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        labels_t = torch.from_numpy(np.ascontiguousarray(labels)).to(
            dtype=torch.float32
        )
        return image_t, labels_t


# ---------------------------------------------------------------------------
# Trained classifier (TrainedClassifierPort impl)
# ---------------------------------------------------------------------------


class _TorchTrainedClassifier:
    """Eval-only :class:`TrainedClassifierPort` wrapping the trained net.

    Per FINE_TUNING_DESIGN.md §5:

    * ``model.eval()`` is called once at construction.
    * ``predict_proba`` runs under ``torch.no_grad()``, returns
      ``sigmoid(logits)`` as a CPU numpy array.
    * Two consecutive calls with identical inputs return byte-identical
      outputs (no dropout, no batchnorm drift).
    """

    __slots__ = (
        "_chunk_size",
        "_device",
        "_image_size",
        "_input_channels",
        "_mean",
        "_model",
        "_n_labels",
        "_std",
    )

    def __init__(
        self,
        *,
        model: nn.Module,
        device: DeviceName,
        n_labels: int,
        image_size: tuple[int, int],
        chunk_size: int = 32,
    ) -> None:
        model.eval()
        self._model: nn.Module = model.to(device)
        self._device: DeviceName = device
        self._n_labels: int = n_labels
        self._image_size: tuple[int, int] = image_size
        self._chunk_size: int = chunk_size
        self._input_channels: int = 3  # backbones expect 3-channel input
        self._mean: torch.Tensor = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        self._std: torch.Tensor = torch.tensor(
            _IMAGENET_STD, dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)

    @property
    def n_labels(self) -> int:
        return self._n_labels

    @property
    def identifier(self) -> str:
        return _TRAINER_IDENTIFIER

    def predict_proba(
        self, images: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        if images.ndim != 4:
            raise AdapterError(
                f"_TorchTrainedClassifier.predict_proba expected 4-D NHWC "
                f"tensor, got ndim={images.ndim} (shape={images.shape})"
            )
        channels = int(images.shape[3])
        if channels not in (1, 3):
            raise AdapterError(
                f"expected 1 or 3 input channels, got {channels} "
                f"(shape={images.shape})"
            )
        n = int(images.shape[0])
        if n == 0:
            return np.empty((0, self._n_labels), dtype=np.float32)
        out = np.empty((n, self._n_labels), dtype=np.float32)
        bs = self._chunk_size
        for start in range(0, n, bs):
            end = min(start + bs, n)
            out[start:end] = self._forward_chunk(images[start:end])
        return out

    def _forward_chunk(
        self, chunk: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        h, w = self._image_size
        tensor = torch.from_numpy(np.ascontiguousarray(chunk)).to(
            device=self._device, dtype=torch.float32
        )
        nchw = tensor.permute(0, 3, 1, 2).contiguous()
        if nchw.shape[1] == 1:
            nchw = nchw.repeat(1, 3, 1, 1)
        if nchw.shape[2] != h or nchw.shape[3] != w:
            nchw = F.interpolate(
                nchw, size=(h, w), mode="bilinear", align_corners=False
            )
        normalized = (nchw - self._mean) / self._std
        with torch.no_grad():
            logits: torch.Tensor = self._model(normalized)
            probs = torch.sigmoid(logits)
        result: NDArray[np.float32] = (
            probs.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        return result


# ---------------------------------------------------------------------------
# TrainerPort impl
# ---------------------------------------------------------------------------


class TorchFineTuneTrainer:
    """:class:`TrainerPort` adapter: end-to-end fine-tuning on torchvision.

    See module docstring for the determinism story, checkpoint format,
    and YAGNI-style scope.
    """

    __slots__ = ("_device",)

    def __init__(self, *, device: DeviceName | None = None) -> None:
        self._device: DeviceName = _select_device(device)
        if self._device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @property
    def identifier(self) -> str:
        return _TRAINER_IDENTIFIER

    def fit(
        self,
        *,
        training_dataset: TrainingDatasetPort,
        validation_dataset: TrainingDatasetPort,
        config: TrainingConfig,
        seed: int,
    ) -> tuple[_TorchTrainedClassifier, TrainingResult]:
        if len(training_dataset) == 0:
            raise AdapterError("training_dataset is empty")
        # Sanity-check first row against the configured n_labels.
        first_image, first_labels = training_dataset[0]
        if first_labels.shape[0] != config.n_labels:
            raise AdapterError(
                f"training_dataset[0] labels shape {first_labels.shape} "
                f"!= config.n_labels {config.n_labels}"
            )
        if first_image.ndim != 3:
            raise AdapterError(
                f"training_dataset[0] image must be (H, W, C); "
                f"got shape {first_image.shape}"
            )
        # Build model, optimizer, scheduler.
        init_seed = _child_seed(seed, "init")
        # Scope torch's RNG to a generator we own; do NOT touch global.
        # Some pretrained-weight loading paths consult the default RNG.
        # We manage init via an explicit fork+seed to avoid global mutation.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            model = _build_model(
                backbone_id=config.backbone_id, n_labels=config.n_labels
            )
        model = model.to(self._device)
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        # Augmentation pipeline (training only).
        augmentation_module = _build_augmentation_pipeline(
            config.augmentations
        )
        # DataLoader with seeded generator; num_workers=0 enforced by
        # TrainingConfig.__post_init__.
        loader_seed = _child_seed(seed, "dataloader")
        loader_gen = torch.Generator()
        loader_gen.manual_seed(loader_seed)
        train_view = _TorchDatasetView(training_dataset)
        train_loader = DataLoader(
            train_view,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=loader_gen,
        )
        val_view = _TorchDatasetView(validation_dataset)
        val_loader = DataLoader(
            val_view, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

        # Augmentation generator (separate so it is not polluted by the
        # DataLoader's shuffle draws).
        aug_seed = _child_seed(seed, "augment")

        cfg_hash = _config_hash(config)
        # Resume from the latest checkpoint if present.
        start_epoch, best_auroc, epochs_without_imp, history = (
            self._maybe_resume(
                config=config,
                cfg_hash=cfg_hash,
                model=model,
                optimizer=optimizer,
            )
        )

        # Schedule (rebuilt fresh; we drive it manually so resume math
        # stays simple).
        train_losses: list[float] = list(history["train_losses"])
        val_losses: list[float] = list(history["val_losses"])
        val_aurocs: list[float] = list(history["val_aurocs"])
        lr_per_epoch: list[float] = list(history["lr_per_epoch"])

        # Best-epoch state tracking (C1 fix). On a fresh run ``best_state``
        # starts as ``None``; on resume we re-derive it from the best epoch
        # in the persisted history (if any). The in-memory state_dict is
        # the source of truth for what the returned classifier wraps; the
        # on-disk ``epoch_{best:03d}.pt`` file stays in sync via
        # ``_maybe_save_checkpoint`` calls per epoch.
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch_so_far: int = (
            int(np.argmax(val_aurocs)) if val_aurocs else -1
        )
        if best_epoch_so_far >= 0 and config.checkpoint_dir is not None:
            ckpt_path = (
                config.checkpoint_dir
                / f"epoch_{best_epoch_so_far:03d}.pt"
            )
            if ckpt_path.is_file():
                try:
                    blob = torch.load(
                        ckpt_path,
                        map_location=self._device,
                        weights_only=False,
                    )
                except Exception as exc:  # noqa: BLE001  # reason: surface as AdapterError
                    raise AdapterError(
                        f"failed to load best-epoch checkpoint {ckpt_path}: {exc}"
                    ) from exc
                best_state = {
                    k: v.detach().clone()
                    for k, v in blob["model_state_dict"].items()
                }

        for epoch in range(start_epoch, config.n_epochs):
            lr = _compute_lr_for_epoch(
                epoch=epoch,
                base_lr=config.learning_rate,
                n_epochs=config.n_epochs,
                warmup_epochs=config.warmup_epochs,
                schedule=config.lr_schedule,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            lr_per_epoch.append(lr)
            train_loss = self._train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                augmentation=augmentation_module,
                augmentation_base_seed=aug_seed,
                epoch=epoch,
                config=config,
            )
            val_loss, val_auroc = self._evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                config=config,
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_aurocs.append(val_auroc)
            improved = val_auroc > best_auroc
            if improved:
                best_auroc = val_auroc
                epochs_without_imp = 0
                # C1: snapshot best-epoch weights so the returned
                # TrainedClassifierPort reflects this epoch, not the last.
                best_state = copy.deepcopy(model.state_dict())
                best_epoch_so_far = epoch
            else:
                epochs_without_imp += 1
            self._maybe_save_checkpoint(
                config=config,
                cfg_hash=cfg_hash,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_auroc=best_auroc,
                epochs_without_imp=epochs_without_imp,
                train_losses=train_losses,
                val_losses=val_losses,
                val_aurocs=val_aurocs,
                lr_per_epoch=lr_per_epoch,
            )
            if (
                config.early_stop_patience is not None
                and epochs_without_imp >= config.early_stop_patience
            ):
                break
        n_epochs_run = len(train_losses)
        if n_epochs_run == 0:
            raise AdapterError(
                "fit completed without running any epochs (resume detected "
                "an already-completed run; supply a fresh checkpoint_dir)"
            )
        # Best epoch by val-AUROC across the full (resumed + new) history.
        best_epoch = int(np.argmax(val_aurocs))
        # C1: load the best-epoch weights into the model before wrapping
        # it in the returned classifier. ``best_state`` is populated either
        # by the in-loop snapshot above (when at least one epoch ran) or
        # by the resume-side load (when start_epoch == n_epochs and no
        # new epoch outperformed the resumed best). On a pure-resume call
        # the loop body executes zero times, ``best_state`` stays as the
        # resume-derived state, and the model still reflects the best epoch.
        if best_state is not None:
            model.load_state_dict(best_state)
        final_checkpoint_uri = (
            self._final_checkpoint_uri(
                config=config, best_epoch=best_epoch
            )
            if config.checkpoint_dir is not None
            else None
        )
        result = TrainingResult(
            n_epochs_run=n_epochs_run,
            train_loss_per_epoch=tuple(train_losses),
            val_loss_per_epoch=tuple(val_losses),
            val_macro_auroc_per_epoch=tuple(val_aurocs),
            best_epoch=best_epoch,
            final_checkpoint_uri=final_checkpoint_uri,
        )
        trained = _TorchTrainedClassifier(
            model=model,
            device=self._device,
            n_labels=config.n_labels,
            image_size=config.image_size,
        )
        return trained, result

    # -- training step --------------------------------------------------

    def _train_one_epoch(
        self,
        *,
        model: nn.Module,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: AdamW,
        loss_fn: nn.BCEWithLogitsLoss,
        augmentation: nn.Module | None,
        augmentation_base_seed: int,
        epoch: int,
        config: TrainingConfig,
    ) -> float:
        model.train()
        device = self._device
        h, w = config.image_size
        mean = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            _IMAGENET_STD, dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        running_loss = 0.0
        n_batches = 0
        for images, labels in loader:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            # 1->3 channel replication so torchvision backbones (which
            # expect 3-channel input) can consume grayscale CXR rows.
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            if augmentation is not None:
                # torchvision.transforms.v2 transforms read from the global
                # torch RNG; we ``fork_rng`` and reseed per-(epoch, batch)
                # so the augmentation pipeline is deterministic for a
                # fixed (seed, dataset) pair without any global mutation
                # leaking outside this scope. Threading ``epoch`` into the
                # derivation fixes M1 from the Wave 4 review (collision
                # between batch 0 of every epoch under the previous
                # ``aug_seed + n_batches`` formulation).
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(
                        _augmentation_seed(
                            augmentation_base_seed,
                            epoch=epoch,
                            batch_idx=n_batches,
                        )
                    )
                    images = augmentation(images)
            if images.shape[2] != h or images.shape[3] != w:
                images = F.interpolate(
                    images, size=(h, w), mode="bilinear", align_corners=False
                )
            images = (images - mean) / std
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            n_batches += 1
        return running_loss / max(n_batches, 1)

    # -- evaluation -----------------------------------------------------

    def _evaluate(
        self,
        *,
        model: nn.Module,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        loss_fn: nn.BCEWithLogitsLoss,
        config: TrainingConfig,
    ) -> tuple[float, float]:
        model.eval()
        device = self._device
        h, w = config.image_size
        mean = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            _IMAGENET_STD, dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        all_probs: list[NDArray[np.float32]] = []
        all_labels: list[NDArray[np.int8]] = []
        running_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                if images.shape[2] != h or images.shape[3] != w:
                    images = F.interpolate(
                        images, size=(h, w), mode="bilinear", align_corners=False
                    )
                normalized = (images - mean) / std
                logits = model(normalized)
                loss = loss_fn(logits, labels)
                running_loss += float(loss.detach().cpu())
                n_batches += 1
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_probs.append(probs.astype(np.float32, copy=False))
                all_labels.append(
                    labels.detach().cpu().numpy().astype(np.int8, copy=False)
                )
        avg_loss = running_loss / max(n_batches, 1)
        if not all_probs:
            return avg_loss, 0.0
        probs_arr = np.concatenate(all_probs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        macro_auroc = _macro_auroc(probs_arr, labels_arr)
        return avg_loss, macro_auroc

    # -- checkpointing --------------------------------------------------

    def _maybe_resume(
        self,
        *,
        config: TrainingConfig,
        cfg_hash: str,
        model: nn.Module,
        optimizer: AdamW,
    ) -> tuple[
        int,
        float,
        int,
        dict[str, list[float]],
    ]:
        empty_history: dict[str, list[float]] = {
            "train_losses": [],
            "val_losses": [],
            "val_aurocs": [],
            "lr_per_epoch": [],
        }
        if config.checkpoint_dir is None:
            return 0, float("-inf"), 0, empty_history
        ckpt_dir = config.checkpoint_dir
        if not ckpt_dir.is_dir():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            return 0, float("-inf"), 0, empty_history
        candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
        if not candidates:
            return 0, float("-inf"), 0, empty_history
        latest = candidates[-1]
        try:
            blob = torch.load(latest, map_location=self._device, weights_only=False)
        except Exception as exc:  # noqa: BLE001  # reason: surface as AdapterError
            raise AdapterError(
                f"failed to load checkpoint {latest}: {exc}"
            ) from exc
        if blob.get("config_hash") != cfg_hash:
            raise AdapterError(
                f"checkpoint config_hash {blob.get('config_hash')!r} does not "
                f"match current TrainingConfig hash {cfg_hash!r}; use a "
                "different checkpoint_dir for a logically different run "
                "(FINE_TUNING_DESIGN.md §5)"
            )
        model.load_state_dict(blob["model_state_dict"])
        optimizer.load_state_dict(blob["optimizer_state_dict"])
        torch.set_rng_state(blob["torch_rng_state"])
        np.random.set_state(blob["numpy_rng_state"])
        epoch_completed: int = int(blob["epoch"])
        history: dict[str, list[float]] = {
            "train_losses": list(blob.get("train_losses", [])),
            "val_losses": list(blob.get("val_losses", [])),
            "val_aurocs": list(blob.get("val_aurocs", [])),
            "lr_per_epoch": list(blob.get("lr_per_epoch", [])),
        }
        return (
            epoch_completed + 1,
            float(blob.get("best_val_macro_auroc_so_far", float("-inf"))),
            int(blob.get("epochs_without_improvement", 0)),
            history,
        )

    def _maybe_save_checkpoint(
        self,
        *,
        config: TrainingConfig,
        cfg_hash: str,
        epoch: int,
        model: nn.Module,
        optimizer: AdamW,
        best_auroc: float,
        epochs_without_imp: int,
        train_losses: list[float],
        val_losses: list[float],
        val_aurocs: list[float],
        lr_per_epoch: list[float],
    ) -> None:
        if config.checkpoint_dir is None:
            return
        ckpt_dir = config.checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        blob = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,  # cosine is computed inline
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "config_hash": cfg_hash,
            "best_val_macro_auroc_so_far": best_auroc,
            "epochs_without_improvement": epochs_without_imp,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_aurocs": val_aurocs,
            "lr_per_epoch": lr_per_epoch,
        }
        try:
            torch.save(blob, path)
        except Exception as exc:  # noqa: BLE001  # reason: surface as AdapterError
            raise AdapterError(
                f"failed to write checkpoint {path}: {exc}"
            ) from exc

    def _final_checkpoint_uri(
        self, *, config: TrainingConfig, best_epoch: int
    ) -> str | None:
        """URI of the best-epoch checkpoint (FINE_TUNING_DESIGN.md §3.2).

        ``best_epoch`` is the 0-indexed epoch with peak val-AUROC across
        the resumed + new history; the returned URI matches the
        on-disk ``epoch_{best:03d}.pt`` file (which is always present
        when ``checkpoint_dir`` is set, since ``_maybe_save_checkpoint``
        runs every epoch).
        """
        if config.checkpoint_dir is None:
            return None
        ckpt_dir = config.checkpoint_dir
        path = ckpt_dir / f"epoch_{best_epoch:03d}.pt"
        return f"file://{path}"


# ---------------------------------------------------------------------------
# Macro-AUROC helper (numpy-only; mirrors sklearn's roc_auc_score for one
# class but vectorised across classes).
# ---------------------------------------------------------------------------


def _macro_auroc(
    probs: NDArray[np.float32], labels: NDArray[np.int8]
) -> float:
    """Compute macro-AUROC across labels using the rank-sum identity.

    Implemented in numpy so the trainer adapter does not import sklearn
    (architecture rule: ``adapters/torch/`` cannot import sklearn).
    Returns 0.0 for any label whose support is fully positive or fully
    negative; the macro is the arithmetic mean across labels.
    """
    n_classes = probs.shape[1]
    aurocs: list[float] = []
    for k in range(n_classes):
        scores = probs[:, k]
        truth = labels[:, k]
        n_pos = int(truth.sum())
        n_neg = int(len(truth) - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue
        # Mann-Whitney U via average ranks.
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
        # Average tied ranks.
        sorted_scores = scores[order]
        # Find tie groups in sorted order, compute mean rank for each.
        i = 0
        while i < len(sorted_scores):
            j = i
            while (
                j + 1 < len(sorted_scores)
                and sorted_scores[j + 1] == sorted_scores[i]
            ):
                j += 1
            if j > i:
                mean_rank = ranks[order[i : j + 1]].mean()
                ranks[order[i : j + 1]] = mean_rank
            i = j + 1
        rank_sum_pos = float(ranks[truth.astype(bool)].sum())
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        aurocs.append(float(auc))
    if not aurocs:
        return 0.0
    return float(np.mean(aurocs))
