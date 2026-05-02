"""Unit tests for the TXRV (torchxrayvision) backbone adapter.

All tests are marked ``@pytest.mark.torch`` and excluded from the default
suite. They construct the adapter against the real TXRV
``densenet121-res224-nih`` weights (downloaded once, cached under
``~/.torchxrayvision/`` via xrv) so they exercise the full feature
extraction path with the publication-grade weights.

Per CLAUDE.md determinism rules: each adapter receives an explicit ``seed``
that is forwarded to ``torch.manual_seed`` *inside* the adapter constructor,
not globally. Reproducibility is asserted by extracting the same input
twice on the same instance and comparing byte-for-byte.

The chunking story is exercised explicitly: feature extraction must be
batch-invariant -- the result of a single ``extract`` call on ``2*chunk+1``
images must equal the concatenation of three smaller calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.torch.txrv_backbone import (
    TXRVDenseNet121NIHBackbone,
    _select_device,
)
from harness.domain.errors import AdapterError

pytestmark = pytest.mark.torch


def _images(n: int, h: int, w: int, c: int, seed: int = 0) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.random(size=(n, h, w, c), dtype=np.float32)


class TestTXRVDenseNet121NIHBackbone:
    def test_embedding_dim_is_1024(self) -> None:
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        assert bb.embedding_dim == 1024

    def test_identifier_matches_documented_string(self) -> None:
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        assert bb.identifier == "txrv-densenet121-res224-nih"

    def test_extract_returns_n_by_1024_float32(self) -> None:
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        feats = bb.extract(_images(2, 8, 8, 1, seed=1))
        assert feats.shape == (2, 1024)
        assert feats.dtype == np.float32

    def test_extract_accepts_three_channel_input(self) -> None:
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        feats = bb.extract(_images(2, 8, 8, 3, seed=2))
        assert feats.shape == (2, 1024)

    def test_chunking_is_batch_invariant(self) -> None:
        """Feature extraction must yield the same numerical result regardless of chunk size.

        We use a tiny chunk_size=2 and run on 5 images (2*2+1). The result
        must equal the concatenation of three sub-extractions covering rows
        [0:2], [2:4], [4:5]. This guards against a state leak inside the
        chunking loop or a bug where the last partial chunk is dropped.
        """
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu", chunk_size=2)
        images = _images(5, 8, 8, 1, seed=3)
        full = bb.extract(images)
        assert full.shape == (5, 1024)
        piece_a = bb.extract(images[0:2])
        piece_b = bb.extract(images[2:4])
        piece_c = bb.extract(images[4:5])
        np.testing.assert_array_equal(full, np.concatenate([piece_a, piece_b, piece_c], axis=0))

    def test_extract_is_deterministic(self) -> None:
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        images = _images(3, 8, 8, 1, seed=42)
        np.testing.assert_array_equal(bb.extract(images), bb.extract(images))

    def test_extract_rejects_wrong_ndim(self) -> None:
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        bad = np.zeros((3, 8, 8), dtype=np.float32)
        with pytest.raises(AdapterError):
            bb.extract(bad)

    def test_extract_rejects_unsupported_channels(self) -> None:
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        bad = np.zeros((1, 8, 8, 2), dtype=np.float32)
        with pytest.raises(AdapterError):
            bb.extract(bad)

    def test_extract_handles_empty_batch(self) -> None:
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        empty = np.zeros((0, 8, 8, 1), dtype=np.float32)
        out = bb.extract(empty)
        assert out.shape == (0, 1024)
        assert out.dtype == np.float32

    def test_eval_mode_no_parameter_drift(self) -> None:
        """The backbone is eval-only; ``extract`` must not modify any parameter.

        We snapshot the L2 norm of every parameter before extraction and
        confirm it is unchanged afterwards. This catches accidental gradient
        flow (which would update params if an optimiser were ever attached)
        and asserts ``model.eval()`` is honoured (BatchNorm running stats
        also remain frozen because no training-time forward path runs).
        """
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        before = [p.detach().clone() for p in bb._model.parameters()]  # noqa: SLF001
        bb.extract(_images(2, 8, 8, 1, seed=99))
        after = list(bb._model.parameters())  # noqa: SLF001
        assert len(before) == len(after)
        for b, a in zip(before, after, strict=True):
            assert (b == a).all().item(), "parameter changed after extract()"

    def test_device_falls_back_to_cpu_when_overridden(self) -> None:
        bb = TXRVDenseNet121NIHBackbone(seed=0, device="cpu")
        assert bb.device == "cpu"


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
