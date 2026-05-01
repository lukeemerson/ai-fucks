"""Contract tests for :class:`harness.ports.dataset.DatasetPort`.

Per ARCHITECTURE.md section 7.1 every port has an abstract contract test class
asserting *behavior* (shape, invariants, determinism). Concrete adapters
subclass and supply an ``adapter`` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.adapters.fakes.dataset import InMemoryFakeDataset
from harness.adapters.fs.nih_dataset import NIHDataset, NIHDatasetConfig
from harness.domain.errors import HarnessError
from harness.domain.types import Dataset, Sample
from harness.ports.dataset import DatasetPort


def _make_samples(n: int, n_labels: int) -> tuple[Sample, ...]:
    return tuple(
        Sample(
            sample_id=f"s{i}",
            patient_id=f"p{i // 2}",
            image_ref=f"ref://{i}",
            labels=tuple(int((i + j) % 2) for j in range(n_labels)),
            metadata={"view": "PA"},
        )
        for i in range(n)
    )


class DatasetPortContract:
    """Abstract contract; subclasses provide an ``adapter`` fixture."""

    @pytest.fixture
    def adapter(self) -> DatasetPort:
        raise NotImplementedError

    def test_load_returns_dataset(self, adapter: DatasetPort) -> None:
        ds = adapter.load()
        assert isinstance(ds, Dataset)

    def test_load_is_deterministic(self, adapter: DatasetPort) -> None:
        a = adapter.load()
        b = adapter.load()
        assert a.name == b.name
        assert a.label_names == b.label_names
        assert tuple(s.sample_id for s in a.samples) == tuple(
            s.sample_id for s in b.samples
        )

    def test_label_vector_lengths_match_label_names(
        self, adapter: DatasetPort
    ) -> None:
        ds = adapter.load()
        n_labels = len(ds.label_names)
        for s in ds.samples:
            assert len(s.labels) == n_labels

    def test_get_image_bytes_returns_bytes_for_known_ref(
        self, adapter: DatasetPort
    ) -> None:
        ds = adapter.load()
        assert ds.samples, "fixture must supply at least one sample"
        ref = ds.samples[0].image_ref
        result = adapter.get_image_bytes(ref)
        assert isinstance(result, bytes)

    def test_get_image_bytes_is_deterministic(self, adapter: DatasetPort) -> None:
        ds = adapter.load()
        ref = ds.samples[0].image_ref
        assert adapter.get_image_bytes(ref) == adapter.get_image_bytes(ref)

    def test_get_image_bytes_unknown_ref_raises_harness_error(
        self, adapter: DatasetPort
    ) -> None:
        with pytest.raises(HarnessError):
            adapter.get_image_bytes("ref://does-not-exist-xyz")


class TestInMemoryFakeDatasetContract(DatasetPortContract):
    @pytest.fixture
    def adapter(self) -> DatasetPort:
        samples = _make_samples(n=6, n_labels=3)
        return InMemoryFakeDataset(
            name="fake",
            label_names=("a", "b", "c"),
            samples=samples,
        )


class TestNIHDatasetContract(DatasetPortContract):
    """Contract suite against the synthetic NIH fixture."""

    @pytest.fixture
    def adapter(self) -> DatasetPort:
        fixture_root = Path(__file__).parent.parent / "fixtures" / "nih"
        config = NIHDatasetConfig(
            csv_path=fixture_root / "Data_Entry_synthetic.csv",
            images_dir=fixture_root / "images",
            image_size=(8, 8),
            cache_size=4,
            disk_cache_dir=None,
            strict_missing_images=True,
        )
        return NIHDataset(config)
