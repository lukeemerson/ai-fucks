"""Randomness port.

Per ARCHITECTURE.md section 4.9, every random source the harness depends on
flows through :class:`RandomnessPort`. Adapters seed Python's ``random``,
NumPy, and (where applicable) torch in :meth:`seed_all`, derive deterministic
sub-seeds via :meth:`child_seed`, and expose seeded integer draws via
:meth:`integers`.

Determinism contract
--------------------
* :meth:`child_seed` must be a pure function of ``(parent_seed, label)``;
  identical inputs return the same value across processes and Python runs.
* Distinct ``label`` strings under the same parent must produce distinct
  child seeds with high probability (the SHA-256-based fakes/sklearn adapters
  guarantee no collisions for the label set used by the runner).
* :meth:`integers` must be a pure function of ``(low, high, size, seed)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["RandomnessPort"]


@runtime_checkable
class RandomnessPort(Protocol):
    """Seeded randomness source for the harness."""

    def seed_all(self, seed: int) -> None:
        """Seed every random source the adapter is responsible for.

        At minimum this includes Python's :mod:`random` and NumPy's global
        random state; adapters that own torch additionally seed it.
        """
        ...

    def integers(
        self, low: int, high: int, size: int, *, seed: int
    ) -> tuple[int, ...]:
        """Draw ``size`` integers in ``[low, high)`` deterministically."""
        ...

    def child_seed(self, parent_seed: int, label: str) -> int:
        """Derive a deterministic child seed from ``(parent_seed, label)``."""
        ...
