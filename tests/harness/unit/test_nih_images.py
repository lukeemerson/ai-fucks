"""Unit tests for ``harness.adapters.fs.nih_images``.

Covers the in-memory LRU image cache and the NIH PNG loader's two surfaces:

* ``read_bytes`` returns the raw on-disk payload (the bytes pathway used by
  ``DatasetPort.get_image_bytes``).
* ``decode`` returns a deterministic ``float32`` ``(H, W, 1)`` array in
  ``[0.0, 1.0]`` and is backed by the LRU cache.

The fixtures are deterministic 8x8 grayscale PNGs whose pixel values follow

    pixel(i, j) = (patient_id * 32 + follow_up * 8 + i * 4 + j) % 256

so we can assert exact float values when ``image_size`` matches the native
resolution. See ``tests/harness/fixtures/nih/README.md`` for the full schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.fs.nih_images import (
    ImageLoaderConfig,
    LRUImageCache,
    NIHImageLoader,
)
from harness.domain.errors import DataError

FIXTURE_IMAGES_DIR = (
    Path(__file__).parent.parent / "fixtures" / "nih" / "images"
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_native_pixels(patient_id: int, follow_up: int) -> NDArray[np.float32]:
    """Reconstruct the canonical 8x8 fixture image as a normalized float array."""
    raw: NDArray[np.float32] = np.fromfunction(
        lambda i, j: (patient_id * 32 + follow_up * 8 + i * 4 + j) % 256,
        (8, 8),
        dtype=np.int64,
    ).astype(np.float32) / np.float32(255.0)
    reshaped: NDArray[np.float32] = raw.reshape(8, 8, 1)
    return reshaped


def _make_loader(
    tmp_path: Path,
    *,
    image_size: tuple[int, int] = (8, 8),
    cache_size: int = 16,
    images_dir: Path | None = None,
) -> NIHImageLoader:
    config = ImageLoaderConfig(
        images_dir=images_dir if images_dir is not None else FIXTURE_IMAGES_DIR,
        image_size=image_size,
        cache_size=cache_size,
        disk_cache_dir=None,
    )
    return NIHImageLoader(config)


# ---------------------------------------------------------------------------
# LRUImageCache
# ---------------------------------------------------------------------------


class TestLRUImageCache:
    def test_empty_cache_returns_none(self) -> None:
        cache = LRUImageCache(max_size=4)
        assert cache.get("foo") is None
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 1

    def test_put_then_get_returns_same_array(self) -> None:
        cache = LRUImageCache(max_size=4)
        arr = np.zeros((2, 2, 1), dtype=np.float32)
        cache.put("foo", arr)
        got = cache.get("foo")
        assert got is not None
        assert np.array_equal(got, arr)

    def test_capacity_eviction_evicts_oldest(self) -> None:
        cache = LRUImageCache(max_size=2)
        a = np.full((1, 1, 1), 0.1, dtype=np.float32)
        b = np.full((1, 1, 1), 0.2, dtype=np.float32)
        c = np.full((1, 1, 1), 0.3, dtype=np.float32)
        cache.put("A", a)
        cache.put("B", b)
        cache.put("C", c)
        assert cache.get("A") is None  # evicted
        assert cache.get("B") is not None
        assert cache.get("C") is not None
        assert len(cache) == 2

    def test_hit_and_miss_counters(self) -> None:
        cache = LRUImageCache(max_size=4)
        for key in ("A", "B", "C"):
            cache.put(key, np.zeros((1, 1, 1), dtype=np.float32))
        # 3 hits
        cache.get("A")
        cache.get("B")
        cache.get("C")
        # 2 misses
        cache.get("X")
        cache.get("Y")
        assert cache.hits == 3
        assert cache.misses == 2

    def test_recency_update_on_get(self) -> None:
        cache = LRUImageCache(max_size=2)
        a = np.full((1, 1, 1), 0.1, dtype=np.float32)
        b = np.full((1, 1, 1), 0.2, dtype=np.float32)
        c = np.full((1, 1, 1), 0.3, dtype=np.float32)
        cache.put("A", a)
        cache.put("B", b)
        # Access A so B becomes the LRU entry.
        assert cache.get("A") is not None
        cache.put("C", c)
        # B should now be evicted; A still present.
        assert cache.get("B") is None
        assert cache.get("A") is not None
        assert cache.get("C") is not None

    def test_len_reports_current_size(self) -> None:
        cache = LRUImageCache(max_size=8)
        assert len(cache) == 0
        cache.put("A", np.zeros((1, 1, 1), dtype=np.float32))
        assert len(cache) == 1
        cache.put("B", np.zeros((1, 1, 1), dtype=np.float32))
        assert len(cache) == 2


# ---------------------------------------------------------------------------
# NIHImageLoader.read_bytes
# ---------------------------------------------------------------------------


class TestReadBytes:
    def test_reads_existing_fixture(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        payload = loader.read_bytes("00000001_000.png")
        assert isinstance(payload, bytes)
        assert len(payload) > 0

    def test_payload_starts_with_png_magic(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        payload = loader.read_bytes("00000001_000.png")
        assert payload.startswith(PNG_MAGIC)

    def test_missing_file_raises_data_error(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with pytest.raises(DataError) as excinfo:
            loader.read_bytes("does_not_exist.png")
        assert "does_not_exist.png" in str(excinfo.value)


# ---------------------------------------------------------------------------
# NIHImageLoader.decode
# ---------------------------------------------------------------------------


class TestDecode:
    def test_native_size_shape(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path, image_size=(8, 8))
        arr = loader.decode("00000001_000.png")
        assert arr.shape == (8, 8, 1)

    def test_resized_shape_matches_config(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path, image_size=(16, 32))
        arr = loader.decode("00000001_000.png")
        assert arr.shape == (16, 32, 1)

    def test_dtype_is_float32(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        arr = loader.decode("00000001_000.png")
        assert arr.dtype == np.float32

    def test_value_range_within_unit_interval(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path, image_size=(8, 8))
        arr = loader.decode("00000001_000.png")
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0

    def test_resized_value_range_within_unit_interval(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path, image_size=(32, 32))
        arr = loader.decode("00000001_000.png")
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0

    def test_decode_is_deterministic(self, tmp_path: Path) -> None:
        loader_a = _make_loader(tmp_path)
        loader_b = _make_loader(tmp_path)
        arr_a = loader_a.decode("00000001_000.png")
        arr_b = loader_b.decode("00000001_000.png")
        assert np.array_equal(arr_a, arr_b)

    def test_second_call_is_cache_hit(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        loader.decode("00000001_000.png")
        loader.decode("00000001_000.png")
        assert loader.cache.hits == 1
        # First call missed (had to load from disk).
        assert loader.cache.misses == 1

    def test_distinct_filenames_have_separate_cache_slots(
        self, tmp_path: Path
    ) -> None:
        loader = _make_loader(tmp_path)
        loader.decode("00000001_000.png")
        loader.decode("00000003_002.png")
        # Two distinct misses; no hits yet.
        assert loader.cache.misses == 2
        assert loader.cache.hits == 0
        # And both entries live in the cache simultaneously.
        assert len(loader.cache) == 2

    def test_native_pixel_values_match_formula(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path, image_size=(8, 8))
        arr = loader.decode("00000001_000.png")
        expected = _expected_native_pixels(patient_id=1, follow_up=0)
        np.testing.assert_allclose(arr, expected, atol=1e-6)

    def test_native_pixel_values_distinct_image(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path, image_size=(8, 8))
        arr = loader.decode("00000003_002.png")
        expected = _expected_native_pixels(patient_id=3, follow_up=2)
        np.testing.assert_allclose(arr, expected, atol=1e-6)

    def test_different_files_produce_different_arrays(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path, image_size=(8, 8))
        a = loader.decode("00000001_000.png")
        b = loader.decode("00000003_002.png")
        assert not np.array_equal(a, b)

    def test_missing_file_raises_data_error(self, tmp_path: Path) -> None:
        loader = _make_loader(tmp_path)
        with pytest.raises(DataError) as excinfo:
            loader.decode("does_not_exist.png")
        assert "does_not_exist.png" in str(excinfo.value)
