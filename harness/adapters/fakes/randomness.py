"""Deterministic fake :class:`~harness.ports.randomness.RandomnessPort`.

Wraps :class:`numpy.random.Generator` and derives child seeds via SHA-256 over
``f"{parent_seed}:{label}"`` truncated to 63 bits. This gives a pure function
``(parent_seed, label) -> child_seed`` with no collisions for the runner's
label set and a uniform distribution over the integer range.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["SeededRandomness"]


_MASK_63 = (1 << 63) - 1


class SeededRandomness:
    """Numpy-backed seeded randomness, deterministic in all entry points."""

    __slots__ = ()

    def seed_all(self, seed: int) -> None:
        """No-op seeding hook required by :class:`RandomnessPort`.

        This adapter manages its own :class:`numpy.random.Generator` instances
        on demand (via :meth:`integers` and :meth:`generator`) and derives
        child seeds purely from ``(parent_seed, label)``. It deliberately does
        **not** mutate any global random state -- ``numpy.random.seed`` and
        ``random.seed`` are intentionally not called here. Touching globals
        would couple every consumer in the process to this adapter's notion
        of determinism and break the port's "pure function of inputs"
        contract. Subclasses that own additional random sources (e.g. torch)
        may override this hook to seed those sources locally.
        """

    def integers(
        self, low: int, high: int, size: int, *, seed: int
    ) -> tuple[int, ...]:
        rng = np.random.default_rng(seed)
        draws = rng.integers(low=low, high=high, size=size)
        return tuple(int(v) for v in draws.tolist())

    def child_seed(self, parent_seed: int, label: str) -> int:
        digest = hashlib.sha256(
            f"{parent_seed}:{label}".encode()
        ).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        # Mask to 63 bits so the result fits in a signed 64-bit int and stays
        # non-negative.
        return value & _MASK_63

    def generator(self, seed: int) -> np.random.Generator:
        """Return a fresh :class:`numpy.random.Generator` seeded with ``seed``.

        Not part of the port surface; offered as a convenience for unit tests
        and adapter internals that want a numpy generator without re-deriving
        one from a child seed.
        """
        return np.random.default_rng(seed)
