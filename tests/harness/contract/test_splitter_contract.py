"""Contract tests for :class:`harness.ports.splitter.SplitterPort`."""

from __future__ import annotations

import pytest

from harness.adapters.fakes.splitter import DeterministicFakeSplitter
from harness.adapters.sklearn.splitter import IterativeStratifiedPatientSplitter
from harness.domain.types import Dataset, Sample, Split
from harness.ports.splitter import SplitterPort


def _ds(n: int, n_labels: int = 3, patients: int | None = None) -> Dataset:
    n_pat = patients if patients is not None else n
    samples = tuple(
        Sample(
            sample_id=f"s{i}",
            patient_id=f"p{i % n_pat}",
            image_ref=f"ref://{i}",
            labels=tuple(int((i + j) % 2) for j in range(n_labels)),
            metadata={},
        )
        for i in range(n)
    )
    return Dataset(name="ds", label_names=tuple(f"l{j}" for j in range(n_labels)),
                   samples=samples)


class SplitterPortContract:
    @pytest.fixture
    def adapter(self) -> SplitterPort:
        raise NotImplementedError

    def test_split_returns_split(self, adapter: SplitterPort) -> None:
        out = adapter.split(_ds(20), val_fraction=0.2, test_fraction=0.2, seed=7)
        assert isinstance(out, Split)

    def test_split_partitions_indices_disjointly(
        self, adapter: SplitterPort
    ) -> None:
        ds = _ds(20)
        out = adapter.split(ds, val_fraction=0.25, test_fraction=0.25, seed=11)
        train = set(out.train_indices)
        val = set(out.val_indices)
        test = set(out.test_indices)
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)

    def test_split_indices_are_subset_of_range(
        self, adapter: SplitterPort
    ) -> None:
        ds = _ds(20)
        out = adapter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=11)
        all_idx = set(out.train_indices) | set(out.val_indices) | set(out.test_indices)
        assert all_idx.issubset(set(range(len(ds.samples))))

    def test_split_covers_all_samples(self, adapter: SplitterPort) -> None:
        ds = _ds(20)
        out = adapter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=11)
        all_idx = set(out.train_indices) | set(out.val_indices) | set(out.test_indices)
        assert all_idx == set(range(len(ds.samples)))

    def test_split_is_deterministic_for_same_seed(
        self, adapter: SplitterPort
    ) -> None:
        ds = _ds(30)
        a = adapter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=42)
        b = adapter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=42)
        assert a.train_indices == b.train_indices
        assert a.val_indices == b.val_indices
        assert a.test_indices == b.test_indices

    def test_split_changes_with_different_seed(
        self, adapter: SplitterPort
    ) -> None:
        ds = _ds(30)
        a = adapter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=1)
        b = adapter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=2)
        assert (
            a.train_indices != b.train_indices
            or a.val_indices != b.val_indices
            or a.test_indices != b.test_indices
        )

    def test_split_respects_patient_grouping(
        self, adapter: SplitterPort
    ) -> None:
        # 30 samples sharing 6 patients (5 samples each) - guarantees that any
        # accidental sample-level split would leak.
        ds = _ds(30, patients=6)
        out = adapter.split(ds, val_fraction=0.34, test_fraction=0.34, seed=99)

        def patients_of(idxs: tuple[int, ...]) -> set[str]:
            return {ds.samples[i].patient_id for i in idxs}

        train_p = patients_of(out.train_indices)
        val_p = patients_of(out.val_indices)
        test_p = patients_of(out.test_indices)
        assert not (train_p & val_p)
        assert not (train_p & test_p)
        assert not (val_p & test_p)

    def test_split_seed_propagates_into_result(
        self, adapter: SplitterPort
    ) -> None:
        ds = _ds(20)
        out = adapter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=123)
        assert out.seed == 123


class TestDeterministicFakeSplitterContract(SplitterPortContract):
    @pytest.fixture
    def adapter(self) -> SplitterPort:
        return DeterministicFakeSplitter()


class TestIterativeStratifiedPatientSplitterContract(SplitterPortContract):
    @pytest.fixture
    def adapter(self) -> SplitterPort:
        return IterativeStratifiedPatientSplitter()
