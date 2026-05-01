"""Splitter port.

Per ARCHITECTURE.md section 4.2, ``SplitterPort`` produces patient-level,
multi-label-stratified train/val/test splits. The signature is deliberately
keyword-only after ``dataset`` to make call sites self-documenting and to keep
fraction/seed wiring explicit at the composition root.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.domain.types import Dataset, Split

__all__ = ["SplitterPort"]


@runtime_checkable
class SplitterPort(Protocol):
    """Produces patient-level, multi-label-stratified splits."""

    def split(
        self,
        dataset: Dataset,
        *,
        val_fraction: float,
        test_fraction: float,
        seed: int,
    ) -> Split:
        """Return a :class:`~harness.domain.types.Split` for ``dataset``.

        Contract:

        * Train/val/test indices are pairwise disjoint.
        * Their union is a subset of ``range(len(dataset.samples))``.
        * No patient appears in more than one of the three index sets.
        * The result is fully determined by ``(dataset, val_fraction,
          test_fraction, seed)``.
        """
        ...
