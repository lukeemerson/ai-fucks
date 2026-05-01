"""Iterative-stratification patient-level :class:`SplitterPort` adapter.

Implements the Sechidis et al. (2011) iterative stratification algorithm at
the patient level, in pure numpy. Patients (not samples) are the atomic unit
of assignment, so a patient never appears in more than one of train/val/test.

Algorithm
---------
1. Group samples by ``patient_id`` and aggregate label vectors via max-pool
   (any positive sample makes the patient positive for that label).
2. Sort patients deterministically by ``patient_id`` and shuffle with a
   :class:`numpy.random.Generator` keyed on ``seed``.
3. Iteratively assign the patient holding the rarest still-needed label to
   the split that needs the most of that label, breaking ties by current
   split size.
4. Expand patient assignments back to sample indices.

This adapter never imports scikit-multilearn; the algorithm is implemented
from first principles.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

from harness.domain.errors import ContractViolation
from harness.domain.types import Dataset, Split

__all__ = ["IterativeStratifiedPatientSplitter"]


_SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


class IterativeStratifiedPatientSplitter:
    """Patient-level multi-label stratified splitter (numpy-only)."""

    __slots__ = ()

    def split(
        self,
        dataset: Dataset,
        *,
        val_fraction: float,
        test_fraction: float,
        seed: int,
    ) -> Split:
        if len(dataset.samples) == 0:
            raise ContractViolation(
                "cannot split an empty dataset (no samples)"
            )
        if val_fraction < 0.0 or test_fraction < 0.0:
            raise ContractViolation(
                "val_fraction and test_fraction must be non-negative"
            )
        if val_fraction + test_fraction >= 1.0:
            raise ContractViolation(
                "val_fraction + test_fraction must be < 1, "
                f"got {val_fraction + test_fraction}"
            )

        # Deterministic patient ordering and aggregate labels.
        patient_ids, patient_labels, patient_to_sample_indices = (
            _aggregate_patients(dataset)
        )
        n_patients = patient_ids.shape[0]
        n_labels = patient_labels.shape[1]

        # Deterministic shuffle keyed on the seed.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_patients)
        patient_ids = patient_ids[perm]
        patient_labels = patient_labels[perm]
        patient_to_sample_indices = [
            patient_to_sample_indices[int(i)] for i in perm
        ]

        # Compute remaining-label budgets per split.
        target_fractions = np.asarray(
            [
                1.0 - val_fraction - test_fraction,
                val_fraction,
                test_fraction,
            ],
            dtype=np.float64,
        )

        assignments = _iterative_stratify(
            patient_labels=patient_labels,
            target_fractions=target_fractions,
            n_labels=n_labels,
        )

        # Build sample-index lists per split, preserving the original sample
        # order within each patient.
        bucketed: list[list[int]] = [[], [], []]
        for patient_idx, split_idx in enumerate(assignments):
            bucketed[int(split_idx)].extend(
                patient_to_sample_indices[patient_idx]
            )
        for bucket in bucketed:
            bucket.sort()

        return Split(
            train_indices=tuple(bucketed[0]),
            val_indices=tuple(bucketed[1]),
            test_indices=tuple(bucketed[2]),
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggregate_patients(
    dataset: Dataset,
) -> tuple[NDArray[np.str_], NDArray[np.int64], list[list[int]]]:
    """Group samples by patient and max-pool their label vectors.

    Returns
    -------
    patient_ids
        ``(n_patients,)`` array of patient ids in deterministic ASCII order.
    patient_labels
        ``(n_patients, n_labels)`` int64 multi-hot matrix.
    patient_to_sample_indices
        For each patient (aligned with ``patient_ids``), the list of original
        sample indices in their order of appearance in ``dataset.samples``.
    """
    n_labels = len(dataset.label_names)
    by_patient: OrderedDict[str, list[int]] = OrderedDict()
    label_acc: dict[str, list[int]] = {}

    for idx, sample in enumerate(dataset.samples):
        if len(sample.labels) != n_labels:
            raise ContractViolation(
                f"sample {sample.sample_id!r}: labels length "
                f"{len(sample.labels)} != n_labels {n_labels}"
            )
        pid = sample.patient_id
        if pid not in by_patient:
            by_patient[pid] = []
            label_acc[pid] = [0] * n_labels
        by_patient[pid].append(idx)
        cur = label_acc[pid]
        for j, v in enumerate(sample.labels):
            if v and not cur[j]:
                cur[j] = 1

    # Deterministic sort by patient_id (lexicographic).
    sorted_pids = sorted(by_patient.keys())
    patient_ids = np.asarray(sorted_pids, dtype=np.str_)
    patient_labels = np.asarray(
        [label_acc[pid] for pid in sorted_pids], dtype=np.int64
    )
    patient_to_sample_indices = [by_patient[pid] for pid in sorted_pids]
    return patient_ids, patient_labels, patient_to_sample_indices


def _iterative_stratify(
    *,
    patient_labels: NDArray[np.int64],
    target_fractions: NDArray[np.float64],
    n_labels: int,
) -> NDArray[np.int64]:
    """Assign each patient to one of three splits via iterative stratification.

    Parameters
    ----------
    patient_labels
        ``(n_patients, n_labels)`` multi-hot matrix.
    target_fractions
        Length-3 array summing to 1: ``[train, val, test]``.
    n_labels
        Number of labels (columns of ``patient_labels``).

    Returns
    -------
    assignments
        ``(n_patients,)`` int64 array, values in ``{0, 1, 2}``.
    """
    n_patients = patient_labels.shape[0]
    assignments = np.full(n_patients, fill_value=-1, dtype=np.int64)

    # Total positives per label across all patients.
    label_totals = patient_labels.sum(axis=0).astype(np.float64)

    # Desired number of patients per split (float, used for tie-breaking).
    desired_sizes = target_fractions * float(n_patients)

    # Desired positives per (split, label).
    # shape: (n_splits, n_labels)
    desired_per_split = np.outer(target_fractions, label_totals)

    # Running counters of remaining positives we still want in each split.
    remaining_per_split = desired_per_split.copy()
    remaining_size = desired_sizes.copy()

    unassigned_mask = np.ones(n_patients, dtype=bool)
    # Per-label set of unassigned patient indices that are positive for it.
    label_to_patients: list[set[int]] = [
        set(np.where(patient_labels[:, j] == 1)[0].tolist())
        for j in range(n_labels)
    ]

    # Patients with no positives are scheduled at the end via a fallback queue
    # (sized purely by remaining_size).
    no_positive: list[int] = [
        int(i)
        for i in range(n_patients)
        if patient_labels[i].sum() == 0
    ]

    while True:
        # Determine the rarest still-needed label among labels with remaining
        # positive patients.
        candidate_labels: list[int] = []
        for j in range(n_labels):
            if label_to_patients[j]:
                candidate_labels.append(j)

        if not candidate_labels:
            break

        # Choose the rarest label by current remaining count of unassigned
        # positives. Ties broken by lowest label index for determinism.
        rarities = np.asarray(
            [len(label_to_patients[j]) for j in candidate_labels],
            dtype=np.int64,
        )
        min_count = int(rarities.min())
        rarest_candidates = [
            candidate_labels[i]
            for i, c in enumerate(rarities)
            if int(c) == min_count
        ]
        rarest_label = rarest_candidates[0]

        # Pick one positive patient for this label. Determinism: lowest
        # patient index among the unassigned positives.
        positives = label_to_patients[rarest_label]
        chosen_patient = min(positives)

        # Choose the split with the largest remaining need for this label;
        # tie-break by largest remaining_size; then lowest split index.
        split_idx = _choose_split(
            remaining_per_split[:, rarest_label],
            remaining_size,
        )

        _assign_patient(
            patient_idx=chosen_patient,
            split_idx=split_idx,
            patient_labels=patient_labels,
            assignments=assignments,
            unassigned_mask=unassigned_mask,
            label_to_patients=label_to_patients,
            remaining_per_split=remaining_per_split,
            remaining_size=remaining_size,
        )

    # Fallback: assign label-less patients in input order to whichever split
    # has the largest remaining_size (ties broken deterministically).
    for patient_idx in no_positive:
        if not unassigned_mask[patient_idx]:
            continue
        split_idx = int(np.argmax(remaining_size))
        # Tie-break: prefer lowest split index.
        max_val = float(remaining_size[split_idx])
        for s in range(remaining_size.shape[0]):
            if float(remaining_size[s]) == max_val:
                split_idx = s
                break
        _assign_patient(
            patient_idx=patient_idx,
            split_idx=split_idx,
            patient_labels=patient_labels,
            assignments=assignments,
            unassigned_mask=unassigned_mask,
            label_to_patients=label_to_patients,
            remaining_per_split=remaining_per_split,
            remaining_size=remaining_size,
        )

    # Any patients that still slipped through (shouldn't happen, but safe):
    leftover = np.where(unassigned_mask)[0]
    for patient_idx_arr in leftover:
        patient_idx = int(patient_idx_arr)
        split_idx = int(np.argmax(remaining_size))
        _assign_patient(
            patient_idx=patient_idx,
            split_idx=split_idx,
            patient_labels=patient_labels,
            assignments=assignments,
            unassigned_mask=unassigned_mask,
            label_to_patients=label_to_patients,
            remaining_per_split=remaining_per_split,
            remaining_size=remaining_size,
        )

    return assignments


def _choose_split(
    remaining_for_label: NDArray[np.float64],
    remaining_size: NDArray[np.float64],
) -> int:
    """Pick the split most needing this label; tie-break by remaining size.

    Final tie-breaker is lowest split index for determinism.
    """
    # Primary criterion: maximize remaining demand for this label.
    max_label = float(remaining_for_label.max())
    primary = [
        s
        for s in range(remaining_for_label.shape[0])
        if float(remaining_for_label[s]) == max_label
    ]
    if len(primary) == 1:
        return primary[0]

    # Tie-break: largest remaining_size.
    sizes = np.asarray(
        [remaining_size[s] for s in primary], dtype=np.float64
    )
    max_size = float(sizes.max())
    secondary = [
        primary[i]
        for i, sz in enumerate(sizes)
        if float(sz) == max_size
    ]
    return secondary[0]


def _assign_patient(
    *,
    patient_idx: int,
    split_idx: int,
    patient_labels: NDArray[np.int64],
    assignments: NDArray[np.int64],
    unassigned_mask: NDArray[np.bool_],
    label_to_patients: list[set[int]],
    remaining_per_split: NDArray[np.float64],
    remaining_size: NDArray[np.float64],
) -> None:
    assignments[patient_idx] = split_idx
    unassigned_mask[patient_idx] = False
    remaining_size[split_idx] -= 1.0
    row = patient_labels[patient_idx]
    n_labels = row.shape[0]
    for j in range(n_labels):
        if int(row[j]) == 1:
            label_to_patients[j].discard(patient_idx)
            remaining_per_split[split_idx, j] -= 1.0
