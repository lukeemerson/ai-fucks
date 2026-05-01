"""In-memory fake :class:`~harness.ports.dataset.DatasetPort`.

Holds an immutable list of :class:`~harness.domain.types.Sample` objects and
returns deterministic synthetic image bytes derived from the sample's
``image_ref``. The fake is deliberately self-contained so contract tests can
drive it from a single ``__init__``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from harness.domain.errors import DataError
from harness.domain.types import Dataset, Sample

__all__ = ["InMemoryFakeDataset"]


class InMemoryFakeDataset:
    """Backs :class:`harness.ports.dataset.DatasetPort` with an in-memory list."""

    __slots__ = ("_dataset", "_known_refs")

    def __init__(
        self,
        *,
        name: str,
        label_names: tuple[str, ...],
        samples: Iterable[Sample],
    ) -> None:
        sample_tuple = tuple(samples)
        self._dataset = Dataset(
            name=name, label_names=label_names, samples=sample_tuple
        )
        self._known_refs: frozenset[str] = frozenset(
            s.image_ref for s in sample_tuple
        )

    def load(self) -> Dataset:
        return self._dataset

    def get_image_bytes(self, image_ref: str) -> bytes:
        if image_ref not in self._known_refs:
            raise DataError(f"unknown image_ref: {image_ref!r}")
        # Deterministic synthetic bytes: a SHA-256 of the ref. Suitable for
        # contract tests; real adapters return real image payloads.
        return hashlib.sha256(image_ref.encode("utf-8")).digest()
