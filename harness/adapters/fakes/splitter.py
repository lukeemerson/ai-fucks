"""Deterministic fake :class:`~harness.ports.splitter.SplitterPort`.

Implements a patient-level split using a stable bucketing scheme:

* Each patient is hashed (SHA-256 of ``f"{seed}:{patient_id}"``) into one of
  ten buckets (``0..9``).
* Buckets are assigned to test (lowest), val (next), and train (rest) based
  on the requested ``test_fraction`` and ``val_fraction``.

This keeps the fake deterministic per ``seed`` (different seeds shuffle the
bucketing because the seed enters the hash), guarantees patient-level
grouping, and never imports numpy or sklearn -- it's a pure-stdlib stand-in
for the iterative-stratification splitter that lives in the sklearn adapter.
"""

from __future__ import annotations

import hashlib

from harness.domain.types import Dataset, Split

__all__ = ["DeterministicFakeSplitter"]


_N_BUCKETS = 10


def _bucket(seed: int, patient_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).digest()
    # Use the first 8 bytes as an unsigned int.
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % _N_BUCKETS


class DeterministicFakeSplitter:
    """Patient-level fake splitter using SHA-256 bucketing."""

    __slots__ = ()

    def split(
        self,
        dataset: Dataset,
        *,
        val_fraction: float,
        test_fraction: float,
        seed: int,
    ) -> Split:
        # Number of buckets allocated to test / val. We always allocate at
        # least one bucket per non-empty split, matching the contract that a
        # caller asking for a non-zero fraction gets a non-empty set when
        # there are enough patients to support it.
        n_test_buckets = max(1, round(test_fraction * _N_BUCKETS))
        n_val_buckets = max(1, round(val_fraction * _N_BUCKETS))
        if n_test_buckets + n_val_buckets >= _N_BUCKETS:
            # Degenerate but keep at least one train bucket.
            n_val_buckets = max(1, _N_BUCKETS - n_test_buckets - 1)

        test_buckets = set(range(n_test_buckets))
        val_buckets = set(range(n_test_buckets, n_test_buckets + n_val_buckets))

        train: list[int] = []
        val: list[int] = []
        test: list[int] = []
        for idx, sample in enumerate(dataset.samples):
            b = _bucket(seed, sample.patient_id)
            if b in test_buckets:
                test.append(idx)
            elif b in val_buckets:
                val.append(idx)
            else:
                train.append(idx)

        return Split(
            train_indices=tuple(train),
            val_indices=tuple(val),
            test_indices=tuple(test),
            seed=seed,
        )
