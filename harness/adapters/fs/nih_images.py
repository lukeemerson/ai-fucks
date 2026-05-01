"""NIH ChestX-ray14 image loader and bounded LRU cache.

This module owns the *image* layer of the NIH dataset adapter (per
``harness/adapters/fs/docs/NIH_DATASET_SPEC.md`` section 4.2). It exposes two
parallel surfaces against the same on-disk PNG corpus:

* :meth:`NIHImageLoader.read_bytes` -- returns the raw PNG payload from disk.
  This is the bytes-pathway used by ``DatasetPort.get_image_bytes`` so the
  backbone (sklearn / torch) can own decoding.
* :meth:`NIHImageLoader.decode` -- returns a deterministic
  ``(H, W, 1)`` ``float32`` array in ``[0.0, 1.0]``. Used for cache
  pre-warming and any direct numpy consumer (smoke tests, utilities).

The dataset-level adapter (``NIHDataset``, Wave 3) wires the bytes surface
into the port; the decode surface is part of this module's public API but
not part of the port contract.

Module scope: pure file/image I/O. No CSV, no label encoding, no knowledge
of patient IDs. See section 4.2 of the spec for the import-allow list.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from harness.domain.errors import DataError

__all__ = [
    "ImageLoaderConfig",
    "LRUImageCache",
    "NIHImageLoader",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImageLoaderConfig:
    """Configuration for :class:`NIHImageLoader`.

    Attributes:
        images_dir: Directory containing the flat PNG files. The loader does
            not walk subdirectories.
        image_size: ``(H, W)`` to which decoded images are bilinearly resized.
            Default ``(224, 224)`` matches ResNet50 / DenseNet121 inputs.
        cache_size: Maximum number of decoded ``float32`` arrays to retain in
            the in-memory LRU. ``0`` disables caching.
        disk_cache_dir: Optional ``.npy`` cache directory for cross-run reuse.
            Not exercised by this module's tests; reserved for future wiring.

    The strict-vs-lenient policy on missing PNGs lives one layer up
    (:class:`NIHDataset`); this loader always raises :class:`DataError` on
    a missing or unreadable file, per spec §13.4.
    """

    images_dir: Path
    image_size: tuple[int, int] = (224, 224)
    cache_size: int = 1024
    disk_cache_dir: Path | None = None


# ---------------------------------------------------------------------------
# Bounded LRU cache (in-memory)
# ---------------------------------------------------------------------------


class LRUImageCache:
    """Bounded least-recently-used cache of decoded image arrays.

    Keyed by string (the absolute path or filename of the PNG) so the cache
    can be shared between ``read_bytes`` and ``decode`` paths if a future
    consumer wants byte caching too. Values are immutable from the cache's
    perspective: the cache stores whatever array it was handed and does not
    copy on read.

    Eviction policy is classic LRU implemented via
    :class:`collections.OrderedDict.move_to_end`. Hits and misses are tracked
    via private counters exposed as read-only properties (used by tests and
    any future observability hook).

    The cache is *not* thread-safe; per spec section 13.8 the adapter is
    single-threaded.
    """

    __slots__ = ("_data", "_hits", "_max_size", "_misses")

    def __init__(self, max_size: int) -> None:
        if max_size < 0:
            msg = f"max_size must be >= 0, got {max_size}"
            raise ValueError(msg)
        self._max_size = max_size
        self._data: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> NDArray[np.float32] | None:
        """Return the cached array for ``key`` or ``None`` on miss.

        On a hit the entry is moved to the most-recently-used end. Counters
        are updated regardless of size; a zero-size cache simply records
        every access as a miss.
        """
        value = self._data.get(key)
        if value is None:
            self._misses += 1
            return None
        self._data.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: str, value: NDArray[np.float32]) -> None:
        """Insert or refresh ``key``, evicting LRU entries to fit ``max_size``.

        With ``max_size == 0`` the call is a no-op so callers can flip the
        cache off via configuration without branching at every put-site.
        """
        if self._max_size == 0:
            return
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = value
            return
        self._data[key] = value
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses


# ---------------------------------------------------------------------------
# NIH image loader
# ---------------------------------------------------------------------------


class NIHImageLoader:
    """Reads NIH PNG files from a flat directory.

    Two entry points share the same ``images_dir`` resolution:

    * :meth:`read_bytes` -- raw on-disk bytes (port pathway).
    * :meth:`decode` -- ``(H, W, 1)`` ``float32`` array in ``[0, 1]``,
      memoised through an :class:`LRUImageCache`.

    Both raise :class:`DataError` on missing or unreadable files; per spec
    §13.4 the loader is the wrong layer for silent-missing semantics. The
    composing :class:`NIHDataset` (Wave 3) is responsible for filtering out
    missing rows when configured to do so.
    """

    __slots__ = ("_cache", "_config")

    def __init__(self, config: ImageLoaderConfig) -> None:
        self._config = config
        self._cache = LRUImageCache(max_size=config.cache_size)

    # ------------------------------------------------------------------ paths
    def _resolve(self, image_index: str) -> Path:
        return self._config.images_dir / image_index

    # ------------------------------------------------------------------ bytes
    def read_bytes(self, image_index: str) -> bytes:
        """Return the raw on-disk PNG bytes for ``image_index``.

        Raises:
            DataError: If the file does not exist or cannot be read. The
                missing filename is included in the message; silent-missing
                semantics belong to the dataset-level adapter.
        """
        path = self._resolve(image_index)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            msg = f"NIH image not found: {image_index} (looked at {path})"
            raise DataError(msg) from exc
        except OSError as exc:
            msg = f"NIH image unreadable: {image_index} (looked at {path})"
            raise DataError(msg) from exc

    # ----------------------------------------------------------------- decode
    def decode(self, image_index: str) -> NDArray[np.float32]:
        """Decode, resize, and normalize the PNG to ``(H, W, 1)`` float32.

        Pipeline (per spec section 5.2):

        1. ``Image.open`` via Pillow.
        2. ``convert("L")`` to single-channel grayscale.
        3. Bilinear resize to ``config.image_size``.
        4. Cast to ``float32`` and divide by ``255.0``.
        5. Defensive ``np.clip`` to ``[0, 1]`` (handles edge cases such as
           palette PNGs where Pillow may return slightly out-of-range
           values after resampling).
        6. Reshape to ``(H, W, 1)``.

        The result is cached in :attr:`cache` keyed by ``image_index``; a
        repeat call with the same key returns the previously decoded array
        without touching disk. Note: the cache is keyed by the supplied
        filename token, not the resolved absolute path, because this
        loader's ``images_dir`` is fixed at construction.
        """
        cached = self._cache.get(image_index)
        if cached is not None:
            return cached

        path = self._resolve(image_index)
        try:
            with Image.open(path) as raw:
                grayscale = raw.convert("L")
                resized = grayscale.resize(
                    (self._config.image_size[1], self._config.image_size[0]),
                    resample=Image.Resampling.BILINEAR,
                )
                pixels = np.asarray(resized, dtype=np.uint8)
        except FileNotFoundError as exc:
            msg = f"NIH image not found: {image_index} (looked at {path})"
            raise DataError(msg) from exc
        except UnidentifiedImageError as exc:
            msg = f"NIH image unreadable (not a valid image): {image_index}"
            raise DataError(msg) from exc
        except OSError as exc:
            msg = f"NIH image unreadable: {image_index} (looked at {path})"
            raise DataError(msg) from exc

        normalized = pixels.astype(np.float32) / np.float32(255.0)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        array = normalized.reshape(
            self._config.image_size[0], self._config.image_size[1], 1
        )
        self._cache.put(image_index, array)
        return array

    # ----------------------------------------------------------------- cache
    @property
    def cache(self) -> LRUImageCache:
        """Expose the in-memory LRU cache (read-only handle)."""
        return self._cache
