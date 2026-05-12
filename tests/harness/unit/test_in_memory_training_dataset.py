"""Unit tests for ``_InMemoryTrainingDataset``.

Per FINE_TUNING_DESIGN.md §4.3 the dataset is built by the composition
root from a ``DatasetPort`` plus a split's index list. v1.1 enforces a
hard 20000-row cap at construction time per the approved §10 answer #3
(streaming variant deferred to v1.2).

Written RED-first under TDD discipline.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from harness.composition._finetune_pipeline import _InMemoryTrainingDataset
from harness.domain.errors import AdapterError, ConfigError
from harness.domain.types import Dataset, Sample


class _BytesDataset:
    """Minimal in-memory ``DatasetPort`` for these unit tests.

    Returns deterministic 8x8 grayscale PNG-like byte payloads keyed by
    ``image_ref``. Decoded by the dataset wrapper into ``(H, W, 1) float32``
    in [0, 1].
    """

    def __init__(self, dataset: Dataset, image_bytes: dict[str, bytes]) -> None:
        self._dataset = dataset
        self._image_bytes = image_bytes

    def load(self) -> Dataset:
        return self._dataset

    def get_image_bytes(self, image_ref: str) -> bytes:
        return self._image_bytes[image_ref]


def _make_dataset(n: int, n_labels: int = 2) -> tuple[Dataset, dict[str, bytes]]:
    samples: list[Sample] = []
    image_bytes: dict[str, bytes] = {}
    rng = np.random.default_rng(0)
    for i in range(n):
        ref = f"img://{i}"
        labels = tuple(int(b) for b in rng.integers(0, 2, size=n_labels))
        samples.append(
            Sample(
                sample_id=f"s{i}",
                patient_id=f"p{i // 3}",
                image_ref=ref,
                labels=labels,
                metadata=MappingProxyType({}),
            )
        )
        # Decoded image bytes: 8x8 single-channel uint8 sequence.
        image_bytes[ref] = bytes(rng.integers(0, 256, size=64).tolist())
    ds = Dataset(
        name="fake",
        label_names=tuple(f"L{i}" for i in range(n_labels)),
        samples=tuple(samples),
    )
    return ds, image_bytes


def test_returns_decoded_image_and_labels() -> None:
    ds, image_bytes = _make_dataset(4)
    src = _BytesDataset(ds, image_bytes)
    indices = (0, 2, 3)
    image_size = (8, 8)
    n_labels = 2

    def decoder(_ref: str, blob: bytes) -> np.ndarray:
        # Tests inject a deterministic decode: bytes -> (8, 8, 1) float32 / 255.
        h, w = image_size
        arr = np.frombuffer(blob, dtype=np.uint8)[: h * w].astype(np.float32) / 255.0
        return arr.reshape(h, w, 1)

    td = _InMemoryTrainingDataset(
        source=src,
        indices=indices,
        n_labels=n_labels,
        image_size=image_size,
        decoder=decoder,
    )
    assert len(td) == len(indices)
    image, labels = td[0]
    assert image.shape == (8, 8, 1)
    assert image.dtype == np.float32
    assert labels.shape == (n_labels,)
    assert labels.dtype == np.int8


def test_index_out_of_range_raises_index_error() -> None:
    ds, image_bytes = _make_dataset(4)
    src = _BytesDataset(ds, image_bytes)

    def decoder(_ref: str, blob: bytes) -> np.ndarray:
        return np.zeros((8, 8, 1), dtype=np.float32)

    td = _InMemoryTrainingDataset(
        source=src,
        indices=(0, 1),
        n_labels=2,
        image_size=(8, 8),
        decoder=decoder,
    )
    with pytest.raises(IndexError):
        td[5]


def test_row_count_cap_enforced_at_construction() -> None:
    """20000-row cap raises :class:`ConfigError` per §10 approved answer #3."""
    ds, image_bytes = _make_dataset(2)
    src = _BytesDataset(ds, image_bytes)

    def decoder(_ref: str, blob: bytes) -> np.ndarray:
        return np.zeros((8, 8, 1), dtype=np.float32)

    too_many_indices = tuple(range(20001))
    with pytest.raises(ConfigError):
        _InMemoryTrainingDataset(
            source=src,
            indices=too_many_indices,
            n_labels=2,
            image_size=(8, 8),
            decoder=decoder,
        )


def test_row_count_cap_at_exactly_20000_is_allowed() -> None:
    """Boundary: exactly 20000 rows is permitted; only > 20000 fails."""
    ds, image_bytes = _make_dataset(2)
    src = _BytesDataset(ds, image_bytes)

    def decoder(_ref: str, blob: bytes) -> np.ndarray:
        return np.zeros((8, 8, 1), dtype=np.float32)

    # Construction does not eagerly decode rows whose indices are out of range
    # in the underlying dataset; the test asserts only that the cap check
    # accepts the boundary value. Dataset itself only has 2 samples; the
    # in-memory adapter is permitted to defer underlying-index validation
    # to row access time.
    indices_ok = tuple([0, 1] * 10000)
    td = _InMemoryTrainingDataset(
        source=src,
        indices=indices_ok,
        n_labels=2,
        image_size=(8, 8),
        decoder=decoder,
    )
    assert len(td) == 20000


def test_label_shape_mismatch_raises_adapter_error() -> None:
    """Underlying labels disagreeing with ``n_labels`` raises ``AdapterError``."""
    ds, image_bytes = _make_dataset(4, n_labels=3)
    src = _BytesDataset(ds, image_bytes)

    def decoder(_ref: str, blob: bytes) -> np.ndarray:
        return np.zeros((8, 8, 1), dtype=np.float32)

    td = _InMemoryTrainingDataset(
        source=src,
        indices=(0, 1),
        n_labels=2,  # mismatched (dataset has 3-label rows)
        image_size=(8, 8),
        decoder=decoder,
    )
    with pytest.raises(AdapterError):
        td[0]
