"""Contract tests for :class:`harness.ports.backbone.BackbonePort`.

Per ARCHITECTURE.md section 7.1 every port has an abstract contract test class
asserting behavior (shape, determinism, error surface). Concrete adapters
subclass the abstract class and supply an ``adapter`` fixture.

Note: this contract follows the numpy-array shape variant of the spec --
``extract`` accepts ``NDArray[float32]`` of shape ``(N, H, W, C)`` and returns
``NDArray[float32]`` of shape ``(N, embedding_dim)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.fakes.backbone import IdentityFakeBackbone
from harness.adapters.fs.cached_backbone import CachedBackbone
from harness.domain.errors import AdapterError
from harness.ports.backbone import BackbonePort


def _make_images(
    n: int, h: int, w: int, c: int, seed: int = 0
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.random(size=(n, h, w, c), dtype=np.float32)


class BackbonePortContract:
    """Abstract contract; subclasses provide an ``adapter`` fixture and shape."""

    image_h: int = 4
    image_w: int = 4
    image_c: int = 1

    @pytest.fixture
    def adapter(self) -> BackbonePort:
        raise NotImplementedError

    def test_embedding_dim_is_positive_int(self, adapter: BackbonePort) -> None:
        assert isinstance(adapter.embedding_dim, int)
        assert adapter.embedding_dim > 0

    def test_extract_returns_n_by_embedding_dim(
        self, adapter: BackbonePort
    ) -> None:
        n = 5
        images = _make_images(n, self.image_h, self.image_w, self.image_c)
        feats = adapter.extract(images)
        assert feats.ndim == 2
        assert feats.shape == (n, adapter.embedding_dim)

    def test_extract_is_deterministic(self, adapter: BackbonePort) -> None:
        images = _make_images(3, self.image_h, self.image_w, self.image_c, seed=7)
        a = adapter.extract(images)
        b = adapter.extract(images)
        np.testing.assert_array_equal(a, b)

    def test_extract_preserves_batch_order(self, adapter: BackbonePort) -> None:
        images = _make_images(4, self.image_h, self.image_w, self.image_c, seed=11)
        full = adapter.extract(images)
        # Extracting a single row at a time must equal the matching row of full.
        for i in range(images.shape[0]):
            single = adapter.extract(images[i : i + 1])
            np.testing.assert_array_equal(single[0], full[i])

    def test_extract_rejects_wrong_ndim(self, adapter: BackbonePort) -> None:
        bad = np.zeros((3, 4, 4), dtype=np.float32)  # missing channel dim
        with pytest.raises(AdapterError):
            adapter.extract(bad)

    def test_extract_rejects_empty_axes(self, adapter: BackbonePort) -> None:
        bad = np.zeros((0, self.image_h, self.image_w, self.image_c), dtype=np.float32)
        # Empty batches are an adapter-specific concern; either accept (return
        # shape (0, D)) or raise AdapterError. Both are valid; just ensure no
        # raw exception leaks.
        try:
            out = adapter.extract(bad)
        except AdapterError:
            return
        assert out.shape == (0, adapter.embedding_dim)


class TestIdentityFakeBackboneContract(BackbonePortContract):
    image_h = 4
    image_w = 4
    image_c = 1

    @pytest.fixture
    def adapter(self) -> BackbonePort:
        return IdentityFakeBackbone(
            image_shape=(self.image_h, self.image_w, self.image_c)
        )

    def test_identity_flattens_input(self) -> None:
        backbone = IdentityFakeBackbone(image_shape=(2, 2, 1))
        images = np.array(
            [[[[0.1], [0.2]], [[0.3], [0.4]]]],
            dtype=np.float32,
        )
        feats = backbone.extract(images)
        assert feats.shape == (1, 4)
        np.testing.assert_array_equal(feats[0], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


# ---------------------------------------------------------------------------
# TorchVision adapters (torch-marked; excluded from default suite).
#
# These run the real torchvision networks with random initialization
# (``weights=None``) so they are network-free but still exercise the full
# tensor-conversion + forward pass.
# ---------------------------------------------------------------------------


@pytest.mark.torch
class TestTorchVisionResNet50BackboneContract(BackbonePortContract):
    image_h = 8
    image_w = 8
    image_c = 1

    @pytest.fixture
    def adapter(self) -> BackbonePort:
        from harness.adapters.torch.backbone import TorchVisionResNet50Backbone

        return TorchVisionResNet50Backbone(seed=0, weights=None, device="cpu")


@pytest.mark.torch
class TestTorchVisionDenseNet121BackboneContract(BackbonePortContract):
    image_h = 8
    image_w = 8
    image_c = 1

    @pytest.fixture
    def adapter(self) -> BackbonePort:
        from harness.adapters.torch.backbone import TorchVisionDenseNet121Backbone

        return TorchVisionDenseNet121Backbone(seed=0, weights=None, device="cpu")


# ---------------------------------------------------------------------------
# TXRV (torchxrayvision) NIH-pretrained DenseNet121 backbone.
#
# Network-restricted: this fixture loads the real ``densenet121-res224-nih``
# weights (downloaded once and cached under ``~/.torchxrayvision/`` by the
# library). Marked ``torch`` so it is excluded from the default suite.
# ---------------------------------------------------------------------------


@pytest.mark.torch
class TestTXRVDenseNet121NIHBackboneContract(BackbonePortContract):
    image_h = 8
    image_w = 8
    image_c = 1

    @pytest.fixture
    def adapter(self) -> BackbonePort:
        from harness.adapters.torch.txrv_backbone import TXRVDenseNet121NIHBackbone

        return TXRVDenseNet121NIHBackbone(seed=0, device="cpu")


# ---------------------------------------------------------------------------
# CachedBackbone (filesystem cache wrapping any inner BackbonePort).
#
# Uses the IdentityFakeBackbone as inner so the suite stays in the default
# fast tier (no torch). ``tmp_path`` makes the cache dir test-isolated.
# A small adapter class gives the fake an ``identifier`` (CachedBackbone
# requires one for cache scoping; IdentityFakeBackbone doesn't expose one).
# ---------------------------------------------------------------------------


class _IdentifiedFake:
    """IdentityFakeBackbone wrapper that adds an ``identifier`` property.

    Stays in the contract-test module (not in adapters/fakes/) because it is
    test scaffolding for the cache contract, not a shipping adapter.
    """

    def __init__(self, image_shape: tuple[int, int, int], identifier: str) -> None:
        self._inner = IdentityFakeBackbone(image_shape=image_shape)
        self._identifier = identifier

    @property
    def embedding_dim(self) -> int:
        return self._inner.embedding_dim

    @property
    def identifier(self) -> str:
        return self._identifier

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._inner.extract(images)


class TestCachedBackboneContract(BackbonePortContract):
    image_h = 4
    image_w = 4
    image_c = 1

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> BackbonePort:
        inner = _IdentifiedFake(
            image_shape=(self.image_h, self.image_w, self.image_c),
            identifier="contract-fake",
        )
        return CachedBackbone(inner=inner, cache_dir=tmp_path)
