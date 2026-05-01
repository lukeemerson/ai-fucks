"""Contract tests for :class:`harness.ports.randomness.RandomnessPort`."""

from __future__ import annotations

import pytest

from harness.adapters.fakes.randomness import SeededRandomness
from harness.ports.randomness import RandomnessPort


class RandomnessPortContract:
    @pytest.fixture
    def adapter(self) -> RandomnessPort:
        raise NotImplementedError

    def test_seed_all_does_not_raise(self, adapter: RandomnessPort) -> None:
        # Behavior contract: side effecting; just ensure it doesn't raise.
        adapter.seed_all(0)

    def test_child_seed_is_deterministic_for_same_args(
        self, adapter: RandomnessPort
    ) -> None:
        a = adapter.child_seed(123, "split")
        b = adapter.child_seed(123, "split")
        assert a == b

    def test_child_seed_differs_for_different_labels(
        self, adapter: RandomnessPort
    ) -> None:
        a = adapter.child_seed(123, "split")
        b = adapter.child_seed(123, "head")
        assert a != b

    def test_child_seed_differs_for_different_parents(
        self, adapter: RandomnessPort
    ) -> None:
        a = adapter.child_seed(1, "split")
        b = adapter.child_seed(2, "split")
        assert a != b

    def test_child_seed_returns_non_negative_int(
        self, adapter: RandomnessPort
    ) -> None:
        s = adapter.child_seed(7, "bootstrap")
        assert isinstance(s, int)
        assert s >= 0

    def test_integers_returns_tuple_of_correct_length(
        self, adapter: RandomnessPort
    ) -> None:
        out = adapter.integers(0, 10, size=5, seed=42)
        assert isinstance(out, tuple)
        assert len(out) == 5
        for v in out:
            assert isinstance(v, int)
            assert 0 <= v < 10

    def test_integers_is_deterministic_for_same_seed(
        self, adapter: RandomnessPort
    ) -> None:
        a = adapter.integers(0, 100, size=8, seed=42)
        b = adapter.integers(0, 100, size=8, seed=42)
        assert a == b

    def test_integers_differs_for_different_seed(
        self, adapter: RandomnessPort
    ) -> None:
        a = adapter.integers(0, 1_000_000, size=8, seed=1)
        b = adapter.integers(0, 1_000_000, size=8, seed=2)
        assert a != b


class TestSeededRandomnessContract(RandomnessPortContract):
    @pytest.fixture
    def adapter(self) -> RandomnessPort:
        return SeededRandomness()
