"""Unit tests for :class:`harness.adapters.fs.cached_backbone.CachedBackbone`.

The cache wraps any inner :class:`BackbonePort`, content-addresses each input
image by ``sha256(image.tobytes())``, scopes the file under
``{cache_dir}/{backbone_id}/{sha[:2]}/{sha}.npy``, and writes atomically via
``tmp + replace`` so a crash mid-write never leaves a corrupt ``.npy``.

Tests assert behavior, not implementation: cache hits return the same bytes as
cache misses; the inner backbone is not re-invoked for hits; backbone-id
scoping prevents cross-backbone collisions; identifier requirement is enforced
on the inner adapter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.fakes.backbone import IdentityFakeBackbone
from harness.adapters.fs.cached_backbone import CachedBackbone
from harness.domain.errors import AdapterError

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CountingBackbone:
    """Wraps an inner backbone and counts ``extract`` calls (and rows seen)."""

    def __init__(self, inner: IdentityFakeBackbone, identifier: str = "counting") -> None:
        self._inner = inner
        self._identifier = identifier
        self.calls: int = 0
        self.rows_seen: int = 0

    @property
    def embedding_dim(self) -> int:
        return self._inner.embedding_dim

    @property
    def identifier(self) -> str:
        return self._identifier

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        self.calls += 1
        self.rows_seen += int(images.shape[0])
        return self._inner.extract(images)


class _NoIdentifierBackbone:
    """Backbone without an ``identifier`` property."""

    def __init__(self, inner: IdentityFakeBackbone) -> None:
        self._inner = inner

    @property
    def embedding_dim(self) -> int:
        return self._inner.embedding_dim

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._inner.extract(images)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_IMAGE_SHAPE: tuple[int, int, int] = (4, 4, 1)


def _images(n: int, seed: int = 0) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.random(size=(n, *_IMAGE_SHAPE), dtype=np.float32)


def _make_inner(identifier: str = "test-bb") -> _CountingBackbone:
    return _CountingBackbone(IdentityFakeBackbone(image_shape=_IMAGE_SHAPE), identifier=identifier)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCachedBackboneShape:
    def test_embedding_dim_delegates_to_inner(self, tmp_path: Path) -> None:
        inner = _make_inner()
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        assert cache.embedding_dim == inner.embedding_dim

    def test_identifier_advertises_cache_wrapper(self, tmp_path: Path) -> None:
        inner = _make_inner(identifier="torchvision.resnet50")
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        assert cache.identifier == "cached:torchvision.resnet50"

    def test_extract_returns_n_by_embedding_dim(self, tmp_path: Path) -> None:
        inner = _make_inner()
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        feats = cache.extract(_images(3, seed=1))
        assert feats.shape == (3, inner.embedding_dim)
        assert feats.dtype == np.float32


class TestCachedBackboneHits:
    def test_second_call_does_not_invoke_inner_for_same_images(self, tmp_path: Path) -> None:
        inner = _make_inner()
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        images = _images(4, seed=1)

        first = cache.extract(images)
        misses_after_first = inner.rows_seen
        assert misses_after_first == 4

        second = cache.extract(images)
        # No new rows fed to the inner backbone — every image is a hit.
        assert inner.rows_seen == misses_after_first
        np.testing.assert_array_equal(first, second)

    def test_hit_and_miss_results_are_byte_identical(self, tmp_path: Path) -> None:
        inner_a = _make_inner()
        cache_a = CachedBackbone(inner=inner_a, cache_dir=tmp_path)
        images = _images(2, seed=11)
        miss_features = cache_a.extract(images)

        inner_b = _make_inner()  # fresh inner; cache populated on disk
        cache_b = CachedBackbone(inner=inner_b, cache_dir=tmp_path)
        hit_features = cache_b.extract(images)

        np.testing.assert_array_equal(miss_features, hit_features)
        # Inner B was never asked to compute anything (every key on disk).
        assert inner_b.rows_seen == 0

    def test_partial_overlap_only_recomputes_misses(self, tmp_path: Path) -> None:
        inner = _make_inner()
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        first_batch = _images(2, seed=21)
        cache.extract(first_batch)
        rows_after_first = inner.rows_seen
        assert rows_after_first == 2

        second_batch_new = _images(3, seed=22)
        # Mix: two rows already cached, three new.
        mixed = np.concatenate([first_batch, second_batch_new], axis=0)
        cache.extract(mixed)
        # Only the three novel rows should reach the inner backbone.
        assert inner.rows_seen - rows_after_first == 3


class TestCachedBackboneScoping:
    def test_different_backbone_id_writes_separate_cache_entries(self, tmp_path: Path) -> None:
        inner_a = _make_inner(identifier="bb-a")
        inner_b = _make_inner(identifier="bb-b")
        cache_a = CachedBackbone(inner=inner_a, cache_dir=tmp_path)
        cache_b = CachedBackbone(inner=inner_b, cache_dir=tmp_path)
        images = _images(1, seed=31)

        cache_a.extract(images)
        cache_b.extract(images)

        # Files for each backbone live under their own subtree.
        a_dir = tmp_path / "bb-a"
        b_dir = tmp_path / "bb-b"
        a_files = list(a_dir.rglob("*.npy"))
        b_files = list(b_dir.rglob("*.npy"))
        assert len(a_files) == 1
        assert len(b_files) == 1
        assert inner_a.rows_seen == 1
        assert inner_b.rows_seen == 1  # b cannot reuse a's cache

    def test_different_image_writes_distinct_cache_entry(self, tmp_path: Path) -> None:
        inner = _make_inner(identifier="bb")
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        cache.extract(_images(1, seed=41))
        cache.extract(_images(1, seed=42))
        all_files = list((tmp_path / "bb").rglob("*.npy"))
        assert len(all_files) == 2


class TestCachedBackboneAtomicity:
    def test_no_tmp_file_left_after_successful_write(self, tmp_path: Path) -> None:
        inner = _make_inner(identifier="bb")
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        cache.extract(_images(3, seed=51))
        leftovers = list(tmp_path.rglob("*.tmp"))
        assert leftovers == []


class TestCachedBackboneIdentifierRequirement:
    def test_inner_without_identifier_raises_adapter_error(self, tmp_path: Path) -> None:
        bare = _NoIdentifierBackbone(IdentityFakeBackbone(image_shape=_IMAGE_SHAPE))
        with pytest.raises(AdapterError):
            CachedBackbone(inner=bare, cache_dir=tmp_path)

    def test_inner_with_blank_identifier_raises_adapter_error(self, tmp_path: Path) -> None:
        inner = _make_inner(identifier="")
        with pytest.raises(AdapterError):
            CachedBackbone(inner=inner, cache_dir=tmp_path)


class TestCachedBackbonePropagatesValidation:
    def test_extract_rejects_wrong_ndim(self, tmp_path: Path) -> None:
        inner = _make_inner()
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        bad = np.zeros((3, 4, 4), dtype=np.float32)  # missing channel dim
        with pytest.raises(AdapterError):
            cache.extract(bad)

    def test_empty_batch_returns_zero_by_embedding_dim(self, tmp_path: Path) -> None:
        inner = _make_inner()
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        empty = np.empty((0, *_IMAGE_SHAPE), dtype=np.float32)
        out = cache.extract(empty)
        assert out.shape == (0, inner.embedding_dim)
        # No work delegated.
        assert inner.rows_seen == 0


class TestCachedBackboneCorruptCache:
    """Corrupt cache files must surface as :class:`AdapterError`.

    The cache reads ``.npy`` files via :func:`numpy.load`, which raises
    ``ValueError`` / ``OSError`` / ``EOFError`` on a truncated or otherwise
    malformed file. A bare numpy exception leaks the dependency through the
    adapter boundary; per the project's "no silent failures" rule and the
    ``AdapterError`` contract, the cache must wrap it so callers see a
    :class:`HarnessError` subclass with a path-bearing message they can act on.
    """

    def test_corrupt_cache_file_raises_adapter_error(self, tmp_path: Path) -> None:
        inner = _make_inner(identifier="bb")
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        images = _images(1, seed=61)

        # Populate the cache.
        cache.extract(images)
        npy_files = list((tmp_path / "bb").rglob("*.npy"))
        assert len(npy_files) == 1, f"expected exactly one cache file, got {npy_files}"
        cache_file = npy_files[0]

        # Corrupt the cached payload (not a valid .npy magic header).
        cache_file.write_bytes(b"not a real npy file")

        # Subsequent extract must raise AdapterError, not a bare numpy error.
        with pytest.raises(AdapterError) as excinfo:
            cache.extract(images)

        message = str(excinfo.value)
        assert "corrupt" in message.lower() or str(cache_file) in message

    def test_truncated_cache_file_raises_adapter_error(self, tmp_path: Path) -> None:
        """A zero-byte cache file is also a corrupt-cache failure."""
        inner = _make_inner(identifier="bb")
        cache = CachedBackbone(inner=inner, cache_dir=tmp_path)
        images = _images(1, seed=62)

        cache.extract(images)
        npy_files = list((tmp_path / "bb").rglob("*.npy"))
        assert len(npy_files) == 1
        cache_file = npy_files[0]

        cache_file.write_bytes(b"")

        with pytest.raises(AdapterError):
            cache.extract(images)
