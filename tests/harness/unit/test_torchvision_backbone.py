"""Unit tests for the TorchVision backbone adapters.

All tests are marked ``@pytest.mark.torch`` and excluded from the default
suite. They construct the adapters with ``weights=None`` (random init) so
they are network-free and fast.

Per CLAUDE.md determinism rules: each adapter receives an explicit ``seed``
that is forwarded to ``torch.manual_seed`` *inside* the adapter constructor,
not globally. Reproducibility is asserted by constructing two adapters with
the same seed and checking byte-equal feature outputs.

Located at ``tests/harness/unit/`` (flat) to match the existing layout and
avoid a name collision between a ``tests/harness/adapters/torch/`` package
and the top-level :mod:`torch` package under ``--import-mode=importlib``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.torch.backbone import (
    TorchVisionDenseNet121Backbone,
    TorchVisionResNet50Backbone,
    _select_device,
)
from harness.domain.errors import AdapterError

pytestmark = pytest.mark.torch


def _images(n: int, h: int, w: int, c: int, seed: int = 0) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.random(size=(n, h, w, c), dtype=np.float32)


# ---------------------------------------------------------------------------
# ResNet50
# ---------------------------------------------------------------------------


class TestTorchVisionResNet50Backbone:
    def test_embedding_dim_is_2048(self) -> None:
        bb = TorchVisionResNet50Backbone(seed=0, weights=None, device="cpu")
        assert bb.embedding_dim == 2048

    def test_extract_returns_n_by_2048_float32(self) -> None:
        bb = TorchVisionResNet50Backbone(seed=0, weights=None, device="cpu")
        feats = bb.extract(_images(2, 8, 8, 1, seed=1))
        assert feats.shape == (2, 2048)
        assert feats.dtype == np.float32

    def test_extract_accepts_three_channel_input(self) -> None:
        bb = TorchVisionResNet50Backbone(seed=0, weights=None, device="cpu")
        feats = bb.extract(_images(2, 8, 8, 3, seed=2))
        assert feats.shape == (2, 2048)

    def test_same_seed_same_output(self) -> None:
        images = _images(3, 8, 8, 1, seed=42)
        a = TorchVisionResNet50Backbone(seed=123, weights=None, device="cpu")
        b = TorchVisionResNet50Backbone(seed=123, weights=None, device="cpu")
        np.testing.assert_array_equal(a.extract(images), b.extract(images))

    def test_different_seed_different_output(self) -> None:
        images = _images(3, 8, 8, 1, seed=42)
        a = TorchVisionResNet50Backbone(seed=1, weights=None, device="cpu")
        b = TorchVisionResNet50Backbone(seed=2, weights=None, device="cpu")
        # Random-init networks with different seeds must produce different features.
        assert not np.allclose(a.extract(images), b.extract(images))

    def test_extract_rejects_wrong_ndim(self) -> None:
        bb = TorchVisionResNet50Backbone(seed=0, weights=None, device="cpu")
        bad = np.zeros((3, 8, 8), dtype=np.float32)
        with pytest.raises(AdapterError):
            bb.extract(bad)

    def test_extract_rejects_unsupported_channels(self) -> None:
        bb = TorchVisionResNet50Backbone(seed=0, weights=None, device="cpu")
        bad = np.zeros((1, 8, 8, 2), dtype=np.float32)  # 2 channels not allowed
        with pytest.raises(AdapterError):
            bb.extract(bad)

    def test_device_falls_back_to_cpu_when_overridden(self) -> None:
        bb = TorchVisionResNet50Backbone(seed=0, weights=None, device="cpu")
        assert bb.device == "cpu"

    def test_device_auto_picks_available_accelerator(self) -> None:
        import torch

        bb = TorchVisionResNet50Backbone(seed=0, weights=None)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            assert bb.device == "mps"
        elif torch.cuda.is_available():
            assert bb.device == "cuda"
        else:
            assert bb.device == "cpu"

    def test_one_channel_input_matches_three_channel_replication(self) -> None:
        """The 1-channel replication policy is part of the adapter contract.

        Calling ``extract`` on a grayscale ``(N, H, W, 1)`` batch must produce
        the same features as calling ``extract`` on the equivalent
        ``np.repeat(..., 3, axis=-1)`` batch. This guards against the
        replication path silently being dropped or normalized differently.
        """
        bb = TorchVisionResNet50Backbone(seed=0, weights=None, device="cpu")
        gray = (
            np.linspace(0.0, 1.0, 256, dtype=np.float32)
            .reshape(1, 16, 16, 1)
        )
        rgb = np.repeat(gray, 3, axis=-1)
        feats_gray = bb.extract(gray)
        feats_rgb = bb.extract(rgb)
        np.testing.assert_allclose(feats_gray, feats_rgb, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# DenseNet121
# ---------------------------------------------------------------------------


class TestTorchVisionDenseNet121Backbone:
    def test_embedding_dim_is_1024(self) -> None:
        bb = TorchVisionDenseNet121Backbone(seed=0, weights=None, device="cpu")
        assert bb.embedding_dim == 1024

    def test_extract_returns_n_by_1024_float32(self) -> None:
        bb = TorchVisionDenseNet121Backbone(seed=0, weights=None, device="cpu")
        feats = bb.extract(_images(2, 8, 8, 1, seed=1))
        assert feats.shape == (2, 1024)
        assert feats.dtype == np.float32

    def test_extract_accepts_three_channel_input(self) -> None:
        bb = TorchVisionDenseNet121Backbone(seed=0, weights=None, device="cpu")
        feats = bb.extract(_images(2, 8, 8, 3, seed=2))
        assert feats.shape == (2, 1024)

    def test_same_seed_same_output(self) -> None:
        images = _images(3, 8, 8, 1, seed=42)
        a = TorchVisionDenseNet121Backbone(seed=99, weights=None, device="cpu")
        b = TorchVisionDenseNet121Backbone(seed=99, weights=None, device="cpu")
        np.testing.assert_array_equal(a.extract(images), b.extract(images))

    def test_different_seed_different_output(self) -> None:
        images = _images(3, 8, 8, 1, seed=42)
        a = TorchVisionDenseNet121Backbone(seed=1, weights=None, device="cpu")
        b = TorchVisionDenseNet121Backbone(seed=2, weights=None, device="cpu")
        assert not np.allclose(a.extract(images), b.extract(images))

    def test_extract_rejects_wrong_ndim(self) -> None:
        bb = TorchVisionDenseNet121Backbone(seed=0, weights=None, device="cpu")
        bad = np.zeros((3, 8, 8), dtype=np.float32)
        with pytest.raises(AdapterError):
            bb.extract(bad)

    def test_extract_rejects_unsupported_channels(self) -> None:
        bb = TorchVisionDenseNet121Backbone(seed=0, weights=None, device="cpu")
        bad = np.zeros((1, 8, 8, 4), dtype=np.float32)  # 4 channels not allowed
        with pytest.raises(AdapterError):
            bb.extract(bad)

    def test_device_falls_back_to_cpu_when_overridden(self) -> None:
        bb = TorchVisionDenseNet121Backbone(seed=0, weights=None, device="cpu")
        assert bb.device == "cpu"


# ---------------------------------------------------------------------------
# Device selection helper -- direct unit coverage of validation behaviour.
# ---------------------------------------------------------------------------


class TestSelectDevice:
    def test_cuda_override_raises_when_cuda_unavailable(self) -> None:
        import torch

        if torch.cuda.is_available():
            pytest.skip("cuda is available on this machine; cannot exercise rejection path")
        with pytest.raises(AdapterError, match="cuda"):
            _select_device("cuda")

    def test_unknown_override_raises(self) -> None:
        with pytest.raises(AdapterError):
            _select_device("xyz")  # type: ignore[arg-type]  # reason: testing runtime validation

    def test_cpu_override_always_succeeds(self) -> None:
        assert _select_device("cpu") == "cpu"
