"""TXRV (torchxrayvision) NIH-pretrained DenseNet121 backbone.

Implements :class:`~harness.ports.backbone.BackbonePort` against
``torchxrayvision``'s ``densenet121-res224-nih`` weights -- a DenseNet121
trained on the full NIH ChestX-ray14 corpus. Produces 1024-dim penultimate
features (``model.classifier.in_features == 1024``).

This adapter complements the ImageNet-trained
:class:`~harness.adapters.torch.backbone.TorchVisionDenseNet121Backbone`
by giving the harness a CXR-pretrained embedding source. In an ad-hoc
ablation (``/tmp/txrv_embed_ablation/run.py``, n=4998) swapping ResNet50 +
ImageNet features for these features lifted macro-AUROC from 0.643 to 0.697
with no other pipeline change. *Caveat*: TXRV NIH weights were trained on
the full NIH corpus including any rows used for evaluation here; absolute
AUROC is optimistically biased and only the *delta* vs ImageNet baselines is
defensible. Document this in any model card / paper writeup that uses this
adapter.

Behaviour:

* Input is ``NDArray[np.float32]`` of shape ``(N, H, W, C)`` with ``C in
  {1, 3}`` and values in ``[0, 1]``. Three-channel input is collapsed to
  one channel via mean-across-channels (chest X-rays are inherently
  grayscale; this matches the reference ablation script).
* Bilinear-resizes to ``224 x 224`` (TXRV's expected input size).
* Maps ``[0, 1]`` -> ``[-1024, 1024]`` to match xrv's HU-like normalisation
  convention (``xrv.datasets.normalize(x*255, 255) == x*2048 - 1024``).
* Forwards through ``model.features(x)`` -> ``F.relu`` -> adaptive average
  pool 2d -> flatten, yielding ``(N, 1024)`` penultimate features. (TXRV's
  ``DenseNet.forward`` applies ``F.relu`` before its classifier; we mirror
  that exactly so the features we extract are what the pretrained
  classifier head consumes.)
* Eval-only: ``model.eval()`` is called once at construction;
  ``extract`` runs inside ``torch.no_grad()``. The classifier head is
  replaced with ``nn.Identity`` so ``model.classifier`` no longer
  participates in any forward path -- but we still drive features through
  ``model.features`` directly to keep the pipeline byte-identical with the
  reference ablation.
* Auto-selects a device (``mps`` -> ``cuda`` -> ``cpu``); a caller-supplied
  override that is not present at runtime raises :class:`AdapterError`.
* Determinism: ``torch.manual_seed(seed)`` runs *inside* the constructor,
  never globally. The backbone is eval-only so the seed does not actually
  drive any forward-time randomness, but we mirror the existing
  torchvision adapters' convention.

MPS-safe chunking: DenseNet121's intermediate activations at 224x224 blow
past the MPSGraph ``INT_MAX`` element limit when the runner submits a full
train split (~3000 images) in one call. ``extract`` chunks its input into
batches of ``chunk_size`` (default 32, the same value the ablation script
used) and stitches the per-chunk features. The chunk size is configurable
so unit tests can exercise the chunking loop with tiny tensors.

v1 deviations from the spec (mirroring the existing torch backbone
adapter; see ARCHITECTURE.md §13):

* ``torch.manual_seed(seed)`` mutates global torch RNG state. Acceptable
  here because the seed is consumed only at construction and inference
  runs under ``torch.no_grad()``.
* ``torch.backends.cudnn.deterministic = True`` is set when the chosen
  device is CUDA; MPS does not expose an equivalent flag and runs on
  MPS are not byte-reproducible across devices.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from numpy.typing import NDArray
from torch import nn

from harness.domain.errors import AdapterError

DeviceName = Literal["cpu", "cuda", "mps"]

_DEFAULT_WEIGHTS_ID: Final[str] = "densenet121-res224-nih"
_DEFAULT_CHUNK_SIZE: Final[int] = 32
_INPUT_SIZE: Final[int] = 224
_EMBEDDING_DIM: Final[int] = 1024
# Map [0, 1] -> [-1024, 1024]; matches xrv.datasets.normalize(x*255, 255).
_TXRV_NORM_SCALE: Final[float] = 2048.0
_TXRV_NORM_OFFSET: Final[float] = 1024.0


def _select_device(override: DeviceName | None) -> DeviceName:
    """Pick a device, preferring MPS, then CUDA, then CPU.

    Mirrors :func:`harness.adapters.torch.backbone._select_device` (kept as a
    private copy here rather than imported to keep the txrv adapter
    self-contained -- the two functions are intentionally identical).
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


def _validate_image_batch(images: NDArray[np.float32]) -> None:
    """Raise :class:`AdapterError` if ``images`` is not a valid NHWC batch."""
    if images.ndim != 4:
        raise AdapterError(
            f"expected 4-D NHWC tensor, got ndim={images.ndim} (shape={images.shape})"
        )
    channels = int(images.shape[3])
    if channels not in (1, 3):
        raise AdapterError(
            f"expected 1 or 3 input channels, got {channels} (shape={images.shape})"
        )


class TXRVDenseNet121NIHBackbone:
    """torchxrayvision NIH-pretrained DenseNet121 feature extractor.

    Constructor arguments:

    * ``seed`` -- forwarded to :func:`torch.manual_seed` for parity with the
      torchvision adapters. See module docstring for the determinism story.
    * ``weights_id`` -- TXRV weights identifier; defaults to
      ``"densenet121-res224-nih"`` (the publication-grade NIH-trained
      DenseNet121). Other valid identifiers include
      ``"densenet121-res224-rsna"`` etc.; the adapter accepts any string TXRV
      recognises but ``embedding_dim`` is hard-coded to 1024 (every
      DenseNet121 variant in the TXRV zoo has the same penultimate width).
    * ``device`` -- one of ``"cpu" | "cuda" | "mps"``, or ``None`` for auto.
      A device override that is not present at runtime raises
      :class:`AdapterError`.
    * ``chunk_size`` -- forward-pass batch size cap. Default 32 (matches the
      reference ablation script and works comfortably on 8-16 GB unified
      memory). Tuned to stay below the MPSGraph ``INT_MAX`` activation limit
      for DenseNet121 at 224x224. ``chunk_size <= 0`` raises
      :class:`AdapterError`.

    The instance is eval-only and exposes a stripped-head model:
    ``model.classifier`` is replaced with ``nn.Identity`` (defensive --
    ``forward`` is not used; we drive ``model.features`` directly so the
    extracted features match what TXRV's classifier expects).
    """

    def __init__(
        self,
        *,
        seed: int,
        weights_id: str = _DEFAULT_WEIGHTS_ID,
        device: DeviceName | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        if chunk_size <= 0:
            raise AdapterError(
                f"chunk_size must be a positive int, got {chunk_size!r}"
            )
        torch.manual_seed(seed)
        self._device: DeviceName = _select_device(device)
        if self._device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self._weights_id: str = weights_id
        model = xrv.models.DenseNet(weights=weights_id)
        # Verify the documented embedding dim matches the loaded model. Every
        # DenseNet121 variant in TXRV's zoo has classifier.in_features == 1024
        # but a defensive assertion catches an unexpected variant slipping in
        # via a custom weights_id.
        observed_in_features = int(model.classifier.in_features)
        if observed_in_features != _EMBEDDING_DIM:
            raise AdapterError(
                f"expected classifier.in_features == {_EMBEDDING_DIM}, got "
                f"{observed_in_features} for weights_id={weights_id!r}"
            )
        model.classifier = nn.Identity()
        model.eval()
        # Capture the features sub-module *before* narrowing to ``nn.Module``.
        # ``xrv.models.DenseNet`` is typed ``Any`` (no torchxrayvision stubs)
        # so ``model.features`` reads as ``Any`` here. Once we narrow to
        # ``nn.Module``, ``self._model.features`` would otherwise resolve to
        # the ``Tensor``-typed default-attribute fallback in torch's stubs.
        features_module: nn.Module = model.features
        self._model: nn.Module = model.to(self._device)
        self._features: nn.Module = features_module.to(self._device)
        self._embedding_dim: int = _EMBEDDING_DIM
        self._chunk_size: int = chunk_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> DeviceName:
        return self._device

    @property
    def identifier(self) -> str:
        return f"txrv-{self._weights_id}"

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _forward_chunk(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        tensor = torch.from_numpy(np.ascontiguousarray(chunk)).to(
            device=self._device, dtype=torch.float32
        )
        nchw = tensor.permute(0, 3, 1, 2).contiguous()
        # TXRV DenseNet expects single-channel input; collapse 3-channel
        # input via mean-across-channels (chest X-rays are inherently
        # grayscale; this matches the reference ablation script).
        if nchw.shape[1] == 3:
            nchw = nchw.mean(dim=1, keepdim=True)
        if nchw.shape[2] != _INPUT_SIZE or nchw.shape[3] != _INPUT_SIZE:
            nchw = F.interpolate(
                nchw,
                size=(_INPUT_SIZE, _INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        # Map [0, 1] -> [-1024, 1024] (xrv normalize convention).
        x = nchw * _TXRV_NORM_SCALE - _TXRV_NORM_OFFSET
        with torch.no_grad():
            feat: torch.Tensor = self._features(x)
            feat = F.relu(feat, inplace=True)
            pooled = F.adaptive_avg_pool2d(feat, (1, 1))
            flat = torch.flatten(pooled, 1)
        result: NDArray[np.float32] = (
            flat.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        return result

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        _validate_image_batch(images)
        n = int(images.shape[0])
        if n == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        # Chunk to keep MPS intermediate tensors below INT_MAX. CPU/CUDA
        # paths could afford a larger chunk but we keep one code path for
        # simplicity (and so the chunking is exercised by every unit test).
        out = np.empty((n, self._embedding_dim), dtype=np.float32)
        bs = self._chunk_size
        for start in range(0, n, bs):
            end = min(start + bs, n)
            out[start:end] = self._forward_chunk(images[start:end])
        return out
