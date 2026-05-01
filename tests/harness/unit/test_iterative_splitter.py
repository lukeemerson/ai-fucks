"""Unit tests for :class:`IterativeStratifiedPatientSplitter`.

Written before implementation (red-first). The splitter performs patient-level
multi-label stratified splitting using iterative stratification (Sechidis et
al. 2011) implemented in pure numpy.
"""

from __future__ import annotations

import random

import pytest

from harness.adapters.sklearn.splitter import IterativeStratifiedPatientSplitter
from harness.domain.errors import ContractViolation
from harness.domain.types import Dataset, Sample, Split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dataset(
    *,
    n_patients: int,
    samples_per_patient: int,
    n_labels: int = 4,
    rare_positive_patients: int = 0,
    seed: int = 0,
) -> Dataset:
    """Construct a dataset with deterministic but non-trivial multi-labels.

    Each patient gets the same label vector across all of their samples (the
    splitter aggregates per patient anyway). One label (index ``n_labels-1``)
    is set positive for exactly ``rare_positive_patients`` patients (the rest
    are negative for that label). The other labels follow a fixed pattern of
    the patient index, ensuring multiple labels co-occur.
    """
    rng = random.Random(seed)
    rare_indices = set(rng.sample(range(n_patients), rare_positive_patients))

    samples: list[Sample] = []
    for p in range(n_patients):
        base = [
            int((p + j) % 3 == 0) for j in range(n_labels - 1)
        ] if n_labels > 1 else []
        rare = 1 if p in rare_indices else 0
        labels = (*base, rare)
        for s in range(samples_per_patient):
            samples.append(
                Sample(
                    sample_id=f"s{p}-{s}",
                    patient_id=f"p{p}",
                    image_ref=f"ref://{p}/{s}",
                    labels=labels,
                    metadata={},
                )
            )

    return Dataset(
        name="ds",
        label_names=tuple(f"l{j}" for j in range(n_labels)),
        samples=tuple(samples),
    )


def _patients_of(ds: Dataset, indices: tuple[int, ...]) -> set[str]:
    return {ds.samples[i].patient_id for i in indices}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoPatientLeakage:
    def test_no_patient_appears_in_more_than_one_split(self) -> None:
        ds = _build_dataset(n_patients=20, samples_per_patient=5)
        splitter = IterativeStratifiedPatientSplitter()
        out = splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=7)
        train = _patients_of(ds, out.train_indices)
        val = _patients_of(ds, out.val_indices)
        test = _patients_of(ds, out.test_indices)
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)


class TestApproximateFractionAdherence:
    def test_within_two_patients_of_target(self) -> None:
        ds = _build_dataset(n_patients=100, samples_per_patient=1)
        splitter = IterativeStratifiedPatientSplitter()
        out = splitter.split(ds, val_fraction=0.15, test_fraction=0.15, seed=3)

        train_p = _patients_of(ds, out.train_indices)
        val_p = _patients_of(ds, out.val_indices)
        test_p = _patients_of(ds, out.test_indices)
        assert abs(len(train_p) - 70) <= 2
        assert abs(len(val_p) - 15) <= 2
        assert abs(len(test_p) - 15) <= 2


class TestDeterminism:
    def test_same_seed_yields_identical_split(self) -> None:
        ds = _build_dataset(n_patients=40, samples_per_patient=2)
        splitter = IterativeStratifiedPatientSplitter()
        a = splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=11)
        b = splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=11)
        assert a.train_indices == b.train_indices
        assert a.val_indices == b.val_indices
        assert a.test_indices == b.test_indices

    def test_different_seed_yields_different_split(self) -> None:
        ds = _build_dataset(n_patients=40, samples_per_patient=2)
        splitter = IterativeStratifiedPatientSplitter()
        a = splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=1)
        b = splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=2)
        assert (
            a.train_indices != b.train_indices
            or a.val_indices != b.val_indices
            or a.test_indices != b.test_indices
        )


class TestRareLabelSpread:
    def test_rare_label_reaches_each_split_for_most_seeds(self) -> None:
        # 5 positives in 100 patients on the rare label (last index).
        # With 70/15/15, expected per-split positives are 3.5 / 0.75 / 0.75.
        # Iterative stratification should keep at least one positive in val
        # and test for the vast majority of seeds.
        good = 0
        trials = 25
        for seed in range(trials):
            ds = _build_dataset(
                n_patients=100,
                samples_per_patient=1,
                rare_positive_patients=5,
                seed=seed,
            )
            splitter = IterativeStratifiedPatientSplitter()
            out = splitter.split(
                ds, val_fraction=0.15, test_fraction=0.15, seed=seed
            )

            def positives_in(idxs: tuple[int, ...], dataset: Dataset = ds) -> int:
                return sum(
                    1 for i in idxs if dataset.samples[i].labels[-1] == 1
                )

            n_train = positives_in(out.train_indices)
            n_val = positives_in(out.val_indices)
            n_test = positives_in(out.test_indices)
            if n_train >= 1 and n_val >= 1 and n_test >= 1:
                good += 1

        # With only 5 positives across 3 splits the property should hold for
        # almost every seed; require at least 80%.
        assert good >= int(0.8 * trials), (
            f"rare-label spread held for only {good}/{trials} seeds"
        )


class TestAllSamplesCovered:
    def test_union_equals_all_indices_and_pairwise_disjoint(self) -> None:
        ds = _build_dataset(n_patients=25, samples_per_patient=3)
        splitter = IterativeStratifiedPatientSplitter()
        out = splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=5)
        train = set(out.train_indices)
        val = set(out.val_indices)
        test = set(out.test_indices)
        assert train | val | test == set(range(len(ds.samples)))
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)


class TestOrderIndependence:
    def test_shuffling_input_samples_does_not_change_split_assignments(
        self,
    ) -> None:
        ds = _build_dataset(n_patients=30, samples_per_patient=4)
        splitter = IterativeStratifiedPatientSplitter()
        a = splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=42)

        rng = random.Random(99)
        shuffled_samples = list(ds.samples)
        rng.shuffle(shuffled_samples)
        ds_shuf = Dataset(
            name=ds.name,
            label_names=ds.label_names,
            samples=tuple(shuffled_samples),
        )
        b = splitter.split(
            ds_shuf, val_fraction=0.2, test_fraction=0.2, seed=42
        )

        # The patient assignment must match across the two orderings.
        a_train = {ds.samples[i].patient_id for i in a.train_indices}
        a_val = {ds.samples[i].patient_id for i in a.val_indices}
        a_test = {ds.samples[i].patient_id for i in a.test_indices}
        b_train = {ds_shuf.samples[i].patient_id for i in b.train_indices}
        b_val = {ds_shuf.samples[i].patient_id for i in b.val_indices}
        b_test = {ds_shuf.samples[i].patient_id for i in b.test_indices}
        assert a_train == b_train
        assert a_val == b_val
        assert a_test == b_test


class TestEmptyInputRejected:
    def test_empty_samples_raises_contract_violation(self) -> None:
        ds = Dataset(
            name="empty", label_names=("l0", "l1"), samples=()
        )
        splitter = IterativeStratifiedPatientSplitter()
        with pytest.raises(ContractViolation):
            splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=0)


class TestSinglePatientEdgeCase:
    def test_single_patient_lands_entirely_in_one_split(self) -> None:
        samples = tuple(
            Sample(
                sample_id=f"s{i}",
                patient_id="only",
                image_ref=f"ref://{i}",
                labels=(1, 0, 1),
                metadata={},
            )
            for i in range(10)
        )
        ds = Dataset(
            name="solo", label_names=("a", "b", "c"), samples=samples
        )
        splitter = IterativeStratifiedPatientSplitter()
        out = splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=0)

        non_empty = [
            len(out.train_indices) > 0,
            len(out.val_indices) > 0,
            len(out.test_indices) > 0,
        ]
        assert sum(non_empty) == 1
        assert (
            len(out.train_indices)
            + len(out.val_indices)
            + len(out.test_indices)
        ) == 10


class TestReturnTypeAndSeed:
    def test_returns_split_with_seed_propagated(self) -> None:
        ds = _build_dataset(n_patients=10, samples_per_patient=2)
        splitter = IterativeStratifiedPatientSplitter()
        out = splitter.split(ds, val_fraction=0.2, test_fraction=0.2, seed=33)
        assert isinstance(out, Split)
        assert out.seed == 33
