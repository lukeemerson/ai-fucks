"""Composition-internal wrappers for the v1 publication pipeline.

Background
----------
``harness/composition/runner.py`` (intentionally stable) converts every
:class:`~harness.ports.dataset.DatasetPort` byte blob to a numpy tensor of
shape ``BYTES_IMAGE_SHAPE = (4, 4, 2)`` *before* calling the backbone. That
bytes-to-tensor pipeline was tailored for the fake adapter set: the fake
backbone is happy to consume any ``(N, 4, 4, 2)`` tensor.

A real torchvision backbone, however, requires ``(N, H, W, C)`` input with
``C in {1, 3}``. PNG bytes from :class:`~harness.adapters.fs.nih_dataset.NIHDataset`
must therefore be decoded into a real image tensor *and* delivered to the
torch backbone, all without modifying the runner.

Solution
--------
A pair of wrappers, used only by :func:`harness.composition.factories.build_publication_runner_v1`:

* :class:`_DecodingDataset` -- a :class:`DatasetPort`. ``get_image_bytes(ref)``
  decodes the underlying PNG via :class:`~harness.adapters.fs.nih_images.NIHImageLoader`
  to ``(H, W, 1) float32 [0, 1]``, stores the decoded array in a shared
  side-channel cache keyed by ``sha256(ref)`` (32 bytes), and returns those
  32 bytes as the byte blob. The runner converts those exact 32 bytes into a
  ``(4, 4, 2)`` tensor row that is byte-recoverable in the wrapper backbone.
* :class:`_DecodingBackbone` -- a :class:`BackbonePort`. On
  ``extract(images)`` it recovers the ``sha256(ref)`` key from each row of
  the ``(N, 4, 4, 2)`` tensor (the ``uint8 -> /255.0 -> *255.0 -> round``
  roundtrip is exact in float32; verified at composition time), looks up the
  decoded image in the shared cache, stacks the lookups into ``(N, H, W, 1)``,
  and forwards the result to the wrapped torchvision backbone.

The wrappers are private (``_``-prefixed) and live in the composition layer
because that is the only layer permitted to bridge multiple adapter packages
(``adapters/fs/`` + ``adapters/torch/``). They implement the existing port
protocols verbatim -- no new port surface is introduced.

Hexagonal note
--------------
This module is the cleanest available workaround that respects the constraint
that ``runner.py`` is stable. The decoded-image cache is populated and
consumed in lockstep within a single ``run_experiment`` call, so the lifetime
of the side-channel is bounded by the run. Future work (v1.1) may refactor
the runner to push bytes-to-tensor responsibility into the dataset adapter,
removing the need for this indirection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from harness.adapters.fs.nih_dataset import NIHDataset
from harness.adapters.fs.nih_images import ImageLoaderConfig, NIHImageLoader
from harness.composition.runner import BYTES_IMAGE_SHAPE
from harness.domain.errors import AdapterError, DataError
from harness.domain.types import Dataset
from harness.ports.backbone import BackbonePort

__all__ = [
    "DecodedImageCache",
    "_DecodingBackbone",
    "_DecodingDataset",
    "_SubsettedDataset",
]

# Number of bytes the runner consumes from each blob (4 * 4 * 2). Returning
# exactly this many bytes from ``get_image_bytes`` means
# ``_bytes_to_image_tensor`` skips its sha256-padding fallback and the bytes
# round-trip exactly through the float32 ``/ 255.0`` -> ``* 255.0`` pipeline.
_RUNNER_BYTES: int = (
    BYTES_IMAGE_SHAPE[0] * BYTES_IMAGE_SHAPE[1] * BYTES_IMAGE_SHAPE[2]
)


def _key_from_ref(ref: str) -> bytes:
    """Derive a 32-byte deterministic key from an image reference string.

    sha256 is used for collision resistance across the (typically) ~112k
    references in the full NIH corpus; truncation to 32 bytes is exactly the
    block size the runner samples from the bytes blob.
    """
    return hashlib.sha256(ref.encode("utf-8")).digest()[:_RUNNER_BYTES]


def _key_from_tensor_row(row: NDArray[np.float32]) -> bytes:
    """Recover the 32-byte key from one ``(4, 4, 2)`` row of the runner tensor.

    The runner builds each row via ``frombuffer(blob[:32], uint8) / 255.0``
    in float32. Multiplying by 255.0 and rounding is an exact inverse for
    every uint8 value in ``[0, 255]`` (verified at composition time).
    """
    flat = row.reshape(-1).astype(np.float32, copy=False) * np.float32(255.0)
    return np.round(flat).astype(np.uint8).tobytes()


class DecodedImageCache:
    """Side-channel cache shared between the dataset and backbone wrappers.

    Keys are 32-byte sha256 prefixes derived from ``Sample.image_ref``; values
    are decoded images of shape ``(H, W, 1)`` ``float32`` in ``[0, 1]``. The
    cache is single-process, single-run, and not thread-safe (the harness is
    single-threaded by design).
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: dict[bytes, NDArray[np.float32]] = {}

    def put(self, key: bytes, value: NDArray[np.float32]) -> None:
        if len(key) != _RUNNER_BYTES:
            raise AdapterError(
                f"DecodedImageCache key must be {_RUNNER_BYTES} bytes, "
                f"got {len(key)}"
            )
        self._data[key] = value

    def get(self, key: bytes) -> NDArray[np.float32]:
        value = self._data.get(key)
        if value is None:
            raise AdapterError(
                "DecodedImageCache miss: no decoded image stored for the "
                "supplied key (the dataset wrapper must be called before "
                "the backbone wrapper sees the corresponding tensor row)"
            )
        return value


class _DecodingDataset:
    """:class:`DatasetPort` wrapper that pre-decodes PNG bytes into a cache.

    ``get_image_bytes(ref)`` decodes the underlying PNG to a ``(H, W, 1)``
    ``float32`` array, stores it in the supplied :class:`DecodedImageCache`
    keyed by ``sha256(ref)``, and returns that key as the runner's byte blob.
    """

    __slots__ = ("_cache", "_image_loader", "_inner", "_ref_to_index")

    def __init__(
        self,
        inner: NIHDataset | _SubsettedDataset,
        cache: DecodedImageCache,
        *,
        image_size: tuple[int, int],
    ) -> None:
        self._inner = inner
        self._cache = cache
        # Build a dedicated image loader so we control the resize size --
        # the wrapped ``NIHDataset`` may have been constructed with a
        # different ``image_size`` for its own caching.
        loader_config = ImageLoaderConfig(
            images_dir=inner.images_dir,
            image_size=image_size,
            cache_size=0,  # we maintain our own decoded cache
            disk_cache_dir=None,
        )
        self._image_loader = NIHImageLoader(loader_config)
        # Precompute a ref -> image_index lookup so ``get_image_bytes`` can
        # resolve the loader's filename token without re-walking the dataset.
        ds = inner.load()
        self._ref_to_index: dict[str, str] = {
            s.image_ref: s.sample_id for s in ds.samples
        }

    def load(self) -> Dataset:
        return self._inner.load()

    def get_image_bytes(self, image_ref: str) -> bytes:
        image_index = self._ref_to_index.get(image_ref)
        if image_index is None:
            raise DataError(
                f"unknown image_ref in _DecodingDataset: {image_ref!r}"
            )
        decoded = self._image_loader.decode(image_index)
        key = _key_from_ref(image_ref)
        self._cache.put(key, decoded)
        return key


class _DecodingBackbone:
    """:class:`BackbonePort` wrapper that pulls decoded images from the cache.

    On ``extract(images)`` the wrapper:

    1. Validates the runner-tensor shape is exactly
       ``(N, BYTES_IMAGE_SHAPE)``.
    2. Recovers each row's 32-byte cache key.
    3. Stacks the cached decoded images into ``(N, H, W, 1)``.
    4. Forwards the stacked tensor to the wrapped backbone's ``extract``.
    """

    __slots__ = ("_cache", "_image_size", "_inner")

    def __init__(
        self,
        inner: BackbonePort,
        cache: DecodedImageCache,
        *,
        image_size: tuple[int, int],
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._image_size = image_size

    @property
    def embedding_dim(self) -> int:
        return self._inner.embedding_dim

    @property
    def identifier(self) -> str:
        ident = getattr(self._inner, "identifier", None)
        if isinstance(ident, str) and ident:
            return f"decoding+{ident}"
        return "decoding+unknown"

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        if images.ndim != 4:
            raise AdapterError(
                f"_DecodingBackbone expected 4-D NHWC tensor, "
                f"got ndim={images.ndim} (shape={images.shape})"
            )
        expected_hwc = BYTES_IMAGE_SHAPE
        actual_hwc = (
            int(images.shape[1]),
            int(images.shape[2]),
            int(images.shape[3]),
        )
        if actual_hwc != expected_hwc:
            raise AdapterError(
                f"_DecodingBackbone expected runner image shape "
                f"{expected_hwc}, got {actual_hwc}"
            )
        n = int(images.shape[0])
        if n == 0:
            return np.empty((0, self._inner.embedding_dim), dtype=np.float32)
        h, w = self._image_size
        decoded = np.empty((n, h, w, 1), dtype=np.float32)
        for i in range(n):
            key = _key_from_tensor_row(images[i])
            decoded[i] = self._cache.get(key)
        return self._inner.extract(decoded)


class _SubsettedDataset:
    """:class:`DatasetPort` wrapper exposing only the first ``n`` samples.

    Used by the publication factory's ``n_samples`` parameter to truncate the
    NIH dataset to a fixed pilot size. Sample ordering is preserved (CSV row
    order). Patient-leakage-free: the NIH CSV is grouped by ``patient_id`` so
    truncating the first ``n`` rows never splits a patient across the included
    and excluded ranges, but the truncation may end mid-patient (a patient's
    image block can be partially truncated). Patient-block-aligned truncation
    is deferred to v1.1 ablations.

    ``get_image_bytes`` delegates to the inner dataset for any ``image_ref``
    that was visible in the truncated range; out-of-range refs raise
    :class:`DataError` to surface contract violations early.
    """

    __slots__ = ("_dataset", "_inner", "_known_refs")

    @property
    def images_dir(self) -> Path:
        """Delegate to the wrapped :class:`NIHDataset`'s images directory."""
        return self._inner.images_dir

    def __init__(self, inner: NIHDataset, n: int) -> None:
        if n <= 0:
            raise AdapterError(
                f"_SubsettedDataset requires n > 0, got {n}"
            )
        loaded = inner.load()
        if n >= len(loaded.samples):
            self._dataset = loaded
        else:
            self._dataset = Dataset(
                name=loaded.name,
                label_names=loaded.label_names,
                samples=loaded.samples[:n],
            )
        self._inner = inner
        self._known_refs: frozenset[str] = frozenset(
            s.image_ref for s in self._dataset.samples
        )

    def load(self) -> Dataset:
        return self._dataset

    def get_image_bytes(self, image_ref: str) -> bytes:
        if image_ref not in self._known_refs:
            raise DataError(
                f"image_ref outside subsetted range: {image_ref!r}"
            )
        return self._inner.get_image_bytes(image_ref)
