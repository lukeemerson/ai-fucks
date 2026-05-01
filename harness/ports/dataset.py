"""Dataset port.

Per ARCHITECTURE.md section 4.1, ``DatasetPort`` is the only surface through
which the harness loads sample metadata and resolves image bytes. Backbones
consume bytes, never paths -- adapters resolve ``Sample.image_ref`` via
:meth:`DatasetPort.get_image_bytes`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.domain.types import Dataset

__all__ = ["DatasetPort"]


@runtime_checkable
class DatasetPort(Protocol):
    """Source-of-truth port for dataset loading and image resolution."""

    def load(self) -> Dataset:
        """Return the full dataset (samples + label vocabulary)."""
        ...

    def get_image_bytes(self, image_ref: str) -> bytes:
        """Resolve an opaque ``image_ref`` to raw image bytes.

        Adapters must raise a :class:`harness.domain.errors.HarnessError`
        subclass (typically :class:`~harness.domain.errors.DataError`) if the
        reference is unknown or the underlying resource is unreadable.
        """
        ...
