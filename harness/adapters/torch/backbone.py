"""TorchVision-backed implementations of :class:`harness.ports.backbone.BackbonePort`.

Two adapters ship in v1:

* :class:`TorchVisionResNet50Backbone` -- ``torchvision.models.resnet50`` with
  ``ResNet50_Weights.IMAGENET1K_V2`` (or ``IMAGENET1K_V1`` if V2 is unavailable
  in the installed torchvision version). Penultimate-layer features are
  ``(N, 2048)``.
* :class:`TorchVisionDenseNet121Backbone` -- ``torchvision.models.densenet121``
  with ``DenseNet121_Weights.IMAGENET1K_V1``. Penultimate features are
  ``(N, 1024)``.

Both adapters:

* Are eval-only: ``model.eval()`` is called once at construction; ``extract``
  runs inside ``torch.no_grad()``. There is no training surface.
* Strip the final classification head (``model.fc`` for ResNet,
  ``model.classifier`` for DenseNet) by replacing it with ``nn.Identity``,
  exposing the penultimate features directly.
* Resize ``(N, H, W, C)`` inputs to ``(N, 3, 224, 224)`` via bilinear
  interpolation. Single-channel (grayscale CXR) input is replicated across
  the three input channels; three-channel input is passed through. Other
  channel counts raise ``AdapterError``.
* Apply the standard ImageNet mean/std normalization
  (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``).
* Auto-select a device (``mps`` -> ``cuda`` -> ``cpu``) but accept a caller
  override via the ``device`` constructor argument. Mismatched overrides
  (e.g. ``device="cuda"`` on a machine without CUDA) raise
  :class:`AdapterError` rather than silently falling back.
* Encapsulate determinism: ``torch.manual_seed(seed)`` runs *inside* the
  constructor, never globally; if the chosen device is CUDA, the constructor
  also enables ``torch.backends.cudnn.deterministic`` and disables
  ``cudnn.benchmark``. Two adapters built with the same seed and the same
  ``weights`` value produce byte-identical features for the same input.

The ``weights`` argument is sentinel-overloaded: omit it to get the default
pretrained weights; pass ``weights=None`` explicitly for random init (used by
unit tests so the suite is network-free and fast); or pass a concrete
``*_Weights`` enum member to pin a specific version. The sentinel is a
private :class:`_DefaultWeights` instance, not a bare ``object()``; this
keeps the constructor signature mypy-narrow.

The output dtype is always ``numpy.float32``. Features are detached, moved
to the host, and converted to ``np.ndarray`` at the boundary.

v1 deviation: this adapter calls ``torch.manual_seed(seed)`` in
``__init__``, which mutates global torch RNG state. This is acceptable for
v1 because (1) the seed is consumed only at construction time for
random-weight init paths used by tests, and (2) inference runs under
``torch.no_grad()`` and does not consume RNG. v1.1 may migrate to a
``torch.Generator``-scoped pattern. See ARCHITECTURE.md §13 ("Torch
backbone adapter (v1)") for the full deviation log.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    densenet121,
    resnet50,
)

from harness.domain.errors import AdapterError

DeviceName = Literal["cpu", "cuda", "mps"]

# Standard ImageNet normalization, used by both ResNet50 and DenseNet121
# torchvision presets.
_IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)
_INPUT_SIZE: Final[int] = 224


class _DefaultWeights:
    """Sentinel: caller did not pass ``weights=``; use the adapter's default pretrained weights."""


_USE_DEFAULT_WEIGHTS: Final = _DefaultWeights()


def _select_device(override: DeviceName | None) -> DeviceName:
    """Pick a device, preferring MPS, then CUDA, then CPU.

    Auto-selection (``override is None``) probes both
    ``torch.backends.mps.is_available()`` *and* ``torch.backends.mps.is_built()``
    before claiming MPS, because the former returns ``True`` on Apple Silicon
    builds where MPS support was disabled at compile time.

    A caller-supplied ``override`` is validated against the runtime: requesting
    a device that is not present raises :class:`AdapterError`. Unknown device
    strings (anything other than ``"cpu" | "cuda" | "mps"``) also raise.
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
    """Raise :class:`AdapterError` if ``images`` is not a valid NHWC batch.

    Allowed channel counts are 1 (grayscale) or 3 (RGB); other counts raise.
    """
    if images.ndim != 4:
        raise AdapterError(
            f"expected 4-D NHWC tensor, got ndim={images.ndim} (shape={images.shape})"
        )
    channels = int(images.shape[3])
    if channels not in (1, 3):
        raise AdapterError(
            f"expected 1 or 3 input channels, got {channels} (shape={images.shape})"
        )


def _to_model_input(
    images: NDArray[np.float32],
    *,
    device: DeviceName,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Convert ``(N, H, W, C)`` numpy float32 to ``(N, 3, 224, 224)`` normalized tensor.

    Callers are responsible for short-circuiting the empty-batch case
    (``images.shape[0] == 0``) before invoking this function; both ``extract``
    methods do so. The function therefore assumes ``N >= 1``.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(images)).to(device=device, dtype=torch.float32)
    # NHWC -> NCHW.
    nchw = tensor.permute(0, 3, 1, 2).contiguous()
    # Replicate single-channel input to 3 channels.
    if nchw.shape[1] == 1:
        nchw = nchw.repeat(1, 3, 1, 1)
    resized = F.interpolate(
        nchw,
        size=(_INPUT_SIZE, _INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return (resized - mean) / std


def _default_resnet50_weights() -> ResNet50_Weights | None:
    """V2 if torchvision exposes it, else V1, else ``None``."""
    v2 = getattr(ResNet50_Weights, "IMAGENET1K_V2", None)
    if isinstance(v2, ResNet50_Weights):
        return v2
    v1 = getattr(ResNet50_Weights, "IMAGENET1K_V1", None)
    if isinstance(v1, ResNet50_Weights):
        return v1
    return None


def _default_densenet121_weights() -> DenseNet121_Weights | None:
    """V1 is the only published torchvision preset for DenseNet121."""
    v1 = getattr(DenseNet121_Weights, "IMAGENET1K_V1", None)
    if isinstance(v1, DenseNet121_Weights):
        return v1
    return None


def _resolve_weights_resnet50(
    weights: ResNet50_Weights | None | _DefaultWeights,
) -> ResNet50_Weights | None:
    if isinstance(weights, _DefaultWeights):
        return _default_resnet50_weights()
    return weights


def _resolve_weights_densenet121(
    weights: DenseNet121_Weights | None | _DefaultWeights,
) -> DenseNet121_Weights | None:
    if isinstance(weights, _DefaultWeights):
        return _default_densenet121_weights()
    return weights


class TorchVisionResNet50Backbone:
    """ResNet50 feature extractor over torchvision's pretrained weights.

    Constructor arguments:

    * ``seed`` -- forwarded to :func:`torch.manual_seed` to make
      random-init runs reproducible. **Required** even when ``weights`` is
      a real pretrained set: the seed still controls any residual stochasticity
      (e.g. the resize kernel on some hardware). Note that this mutates global
      torch RNG state; see the module docstring for the v1 deviation rationale.
    * ``weights`` -- a ``ResNet50_Weights`` member, or ``None`` (random init,
      used by tests). When omitted, defaults to ``IMAGENET1K_V2`` if available
      on the installed torchvision version, falling back to ``IMAGENET1K_V1``.
    * ``device`` -- one of ``"cpu" | "cuda" | "mps"``, or ``None`` for auto.
      A device override that is not present at runtime raises
      :class:`AdapterError`.

    The instance is eval-only and exposes a stripped-head model: ``model.fc``
    is replaced with ``nn.Identity`` so ``forward`` returns penultimate
    features of shape ``(N, 2048)``.
    """

    def __init__(
        self,
        *,
        seed: int,
        weights: ResNet50_Weights | None | _DefaultWeights = _USE_DEFAULT_WEIGHTS,
        device: DeviceName | None = None,
    ) -> None:
        torch.manual_seed(seed)
        self._device: DeviceName = _select_device(device)
        if self._device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        resolved_weights = _resolve_weights_resnet50(weights)
        model = resnet50(weights=resolved_weights)
        model.fc = nn.Identity()
        model.eval()
        self._model: nn.Module = model.to(self._device)
        self._embedding_dim: int = 2048
        # Pre-stage normalization tensors on the chosen device.
        self._mean: torch.Tensor = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)
        self._std: torch.Tensor = torch.tensor(
            _IMAGENET_STD, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> DeviceName:
        return self._device

    @property
    def identifier(self) -> str:
        return "torchvision.resnet50"

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        _validate_image_batch(images)
        if images.shape[0] == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)
        x = _to_model_input(images, device=self._device, mean=self._mean, std=self._std)
        with torch.no_grad():
            features = self._model(x)
        result: NDArray[np.float32] = features.detach().cpu().numpy().astype(np.float32, copy=False)
        return result


class TorchVisionDenseNet121Backbone:
    """DenseNet121 feature extractor over torchvision's pretrained weights.

    Identical surface to :class:`TorchVisionResNet50Backbone` except for the
    underlying network and the embedding dimension. ``model.classifier`` is
    replaced with ``nn.Identity`` so ``forward`` returns penultimate features
    of shape ``(N, 1024)``.
    """

    def __init__(
        self,
        *,
        seed: int,
        weights: DenseNet121_Weights | None | _DefaultWeights = _USE_DEFAULT_WEIGHTS,
        device: DeviceName | None = None,
    ) -> None:
        torch.manual_seed(seed)
        self._device: DeviceName = _select_device(device)
        if self._device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        resolved_weights = _resolve_weights_densenet121(weights)
        model = densenet121(weights=resolved_weights)
        model.classifier = nn.Identity()
        model.eval()
        self._model: nn.Module = model.to(self._device)
        self._embedding_dim: int = 1024
        self._mean: torch.Tensor = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)
        self._std: torch.Tensor = torch.tensor(
            _IMAGENET_STD, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> DeviceName:
        return self._device

    @property
    def identifier(self) -> str:
        return "torchvision.densenet121"

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        _validate_image_batch(images)
        if images.shape[0] == 0:
            return np.empty((0, self._embedding_dim), dtype=np.float32)
        x = _to_model_input(images, device=self._device, mean=self._mean, std=self._std)
        with torch.no_grad():
            features = self._model(x)
        result: NDArray[np.float32] = features.detach().cpu().numpy().astype(np.float32, copy=False)
        return result
