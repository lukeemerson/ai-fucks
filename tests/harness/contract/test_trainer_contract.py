"""Contract tests for :class:`harness.ports.trainer.TrainerPort`.

Per ``harness/docs/FINE_TUNING_DESIGN.md`` §6.1 the trainer port has a single
abstract contract base class :class:`TrainerPortContract`. Concrete adapters
land in the v1.1 implementation PR, at which point the implementation agent
appends a ``TestTorchFineTuneTrainerContract(TrainerPortContract)`` subclass
that supplies an ``adapter`` fixture. This design PR ships only the abstract
base; pytest never collects it as a test class because its name does not
start with ``Test``.

The base also bundles a small synthetic :class:`TrainingDatasetPort`
implementation (``_TinyDataset``) used by the contract assertions: 16 rows,
8x8 grayscale, two-class. Class 0 is "dark image" (mean pixel < 0.5),
class 1 is "bright image" (mean pixel >= 0.5). Solvable in 2-3 epochs by
any reasonable trainer; non-trivial enough that
``test_loss_decreases_monotonically`` is a meaningful signal.

The base class exists in this PR so the implementation agent can subclass
it under TDD red-green discipline: write the subclass, watch every
contract assertion fail RED against an empty trainer adapter, then
implement the trainer to GREEN.

Note: imports below reference :class:`TrainingConfig` from
``harness.domain.types``. That type is added by the implementation PR; in
this design PR the import is guarded under ``TYPE_CHECKING`` so the test
module imports cleanly today, while the abstract methods that *use*
``TrainingConfig`` are available for subclasses once the type lands.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.domain.errors import AdapterError
from harness.domain.types import TrainingConfig
from harness.ports.trainer import (
    TrainedClassifierPort,
    TrainerPort,
    TrainingDatasetPort,
)


class _TinyDataset:
    """Two-class synthetic :class:`TrainingDatasetPort` used by the contract.

    16 rows, 8x8 grayscale. Class 0 is "dark" (uniform pixels in [0, 0.5));
    class 1 is "bright" (uniform pixels in [0.5, 1)). Multi-hot labels of
    shape ``(2,)`` so ``n_labels=2``.

    Determinism: rows depend only on the constructor seed; no global RNG.
    """

    def __init__(self, *, seed: int, n_rows: int = 16) -> None:
        rng = np.random.default_rng(seed)
        rows: list[tuple[NDArray[np.float32], NDArray[np.int8]]] = []
        for i in range(n_rows):
            is_bright = bool(i % 2)
            low, high = (0.5, 1.0) if is_bright else (0.0, 0.5)
            image = rng.uniform(low, high, size=(8, 8, 1)).astype(np.float32)
            labels = np.array(
                [int(not is_bright), int(is_bright)], dtype=np.int8
            )
            rows.append((image, labels))
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(
        self, index: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
        return self._rows[index]


class TrainerPortContract:
    """Abstract contract for :class:`TrainerPort` adapters.

    Subclasses MUST override the ``adapter`` and ``training_config`` fixtures.
    The class is intentionally **not** named with a ``Test`` prefix so pytest
    skips collection on the abstract base; concrete subclasses
    (``TestTorchFineTuneTrainerContract`` etc., landing in the implementation
    PR) supply real fixtures and become the actual test suite.
    """

    @pytest.fixture
    def adapter(self) -> TrainerPort:
        raise NotImplementedError(
            "Subclasses must override the ``adapter`` fixture with a real "
            "TrainerPort instance."
        )

    @pytest.fixture
    def training_config(self) -> TrainingConfig:
        raise NotImplementedError(
            "Subclasses must override the ``training_config`` fixture with a "
            "TrainingConfig pinned to ``n_labels=2`` to match _TinyDataset."
        )

    @pytest.fixture
    def training_dataset(self) -> TrainingDatasetPort:
        return _TinyDataset(seed=0)

    @pytest.fixture
    def validation_dataset(self) -> TrainingDatasetPort:
        return _TinyDataset(seed=1)

    # --- shape & type contract -----------------------------------------

    def test_fit_returns_trained_classifier_port(
        self,
        adapter: TrainerPort,
        training_dataset: TrainingDatasetPort,
        validation_dataset: TrainingDatasetPort,
        training_config: TrainingConfig,
    ) -> None:
        trained, _result = adapter.fit(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            config=training_config,
            seed=0,
        )
        assert isinstance(trained, TrainedClassifierPort)
        assert trained.n_labels == 2

    def test_predict_proba_outputs_in_unit_interval(
        self,
        adapter: TrainerPort,
        training_dataset: TrainingDatasetPort,
        validation_dataset: TrainingDatasetPort,
        training_config: TrainingConfig,
    ) -> None:
        trained, _ = adapter.fit(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            config=training_config,
            seed=0,
        )
        # Stack the validation rows into an (N, H, W, C) batch.
        images = np.stack(
            [validation_dataset[i][0] for i in range(len(validation_dataset))]
        ).astype(np.float32)
        probs = trained.predict_proba(images)
        assert probs.shape == (len(validation_dataset), 2)
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    # --- behaviour contract --------------------------------------------

    def test_predict_proba_is_eval_only(
        self,
        adapter: TrainerPort,
        training_dataset: TrainingDatasetPort,
        validation_dataset: TrainingDatasetPort,
        training_config: TrainingConfig,
    ) -> None:
        trained, _ = adapter.fit(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            config=training_config,
            seed=0,
        )
        images = np.stack(
            [validation_dataset[i][0] for i in range(4)]
        ).astype(np.float32)
        a = trained.predict_proba(images)
        b = trained.predict_proba(images)
        np.testing.assert_array_equal(a, b)

    def test_loss_decreases_monotonically(
        self,
        adapter: TrainerPort,
        training_dataset: TrainingDatasetPort,
        validation_dataset: TrainingDatasetPort,
        training_config: TrainingConfig,
    ) -> None:
        """Mean training loss must trend down across epochs.

        Allows small upward jitter (``rtol`` baked into the assertion):
        ``train_loss[-1] < train_loss[0]`` is the load-bearing claim.
        Subclasses with very small ``n_epochs`` may override this to a
        stricter monotonicity check.
        """
        _trained, result = adapter.fit(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            config=training_config,
            seed=0,
        )
        losses = result.train_loss_per_epoch
        assert len(losses) >= 2, (
            "test_loss_decreases_monotonically requires n_epochs >= 2"
        )
        assert losses[-1] < losses[0], (
            f"train loss did not decrease: first={losses[0]}, last={losses[-1]}"
        )

    def test_determinism_byte_identical_predictions(
        self,
        adapter: TrainerPort,
        training_dataset: TrainingDatasetPort,
        validation_dataset: TrainingDatasetPort,
        training_config: TrainingConfig,
    ) -> None:
        """Two trainers fed the same seed produce byte-identical predictions.

        Asserts on ``predict_proba`` output rather than ``state_dict()``
        because ``TrainedClassifierPort`` does not (and should not) expose
        ``state_dict``. Adapters may produce different state-dict layouts
        (e.g. wrapped vs unwrapped models) yet still satisfy the
        port-level determinism contract.
        """
        trained_a, _ = adapter.fit(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            config=training_config,
            seed=0,
        )
        trained_b, _ = adapter.fit(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            config=training_config,
            seed=0,
        )
        images = np.stack(
            [validation_dataset[i][0] for i in range(4)]
        ).astype(np.float32)
        np.testing.assert_array_equal(
            trained_a.predict_proba(images), trained_b.predict_proba(images)
        )

    # --- failure surface -----------------------------------------------

    def test_fit_rejects_empty_training_dataset(
        self,
        adapter: TrainerPort,
        validation_dataset: TrainingDatasetPort,
        training_config: TrainingConfig,
    ) -> None:
        empty = _TinyDataset(seed=0, n_rows=0)
        with pytest.raises(AdapterError):
            adapter.fit(
                training_dataset=empty,
                validation_dataset=validation_dataset,
                config=training_config,
                seed=0,
            )

    def test_fit_rejects_label_shape_mismatch(
        self,
        adapter: TrainerPort,
        validation_dataset: TrainingDatasetPort,
        training_config: TrainingConfig,
    ) -> None:
        """Dataset rows whose labels disagree with ``config.n_labels`` must error."""
        # 3-class labels but config pins n_labels to 2 -> AdapterError.
        rng = np.random.default_rng(0)

        class _MismatchedDataset:
            def __len__(self) -> int:
                return 8

            def __getitem__(
                self, index: int
            ) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
                return (
                    rng.random(size=(8, 8, 1), dtype=np.float32),
                    np.zeros((3,), dtype=np.int8),
                )

        with pytest.raises(AdapterError):
            adapter.fit(
                training_dataset=_MismatchedDataset(),
                validation_dataset=validation_dataset,
                config=training_config,
                seed=0,
            )


# ---------------------------------------------------------------------------
# Concrete subclasses for v1.1.
#
# ``TestTorchFineTuneTrainerContract`` ships the trainer-on-CPU adapter
# against the abstract base's two-class _TinyDataset. Tagged ``torch`` so the
# default fast suite (``-m 'not smoke and not slow and not torch'``) skips
# the heavy DenseNet121 forward+backward pass; opt in with ``pytest -m
# torch``.
# ---------------------------------------------------------------------------


def _torch_finetune_tiny_config() -> TrainingConfig:
    """A TrainingConfig calibrated for ``_TinyDataset`` (16 rows, 8x8x1).

    Uses CPU + 32x32 resize + constant LR + no augmentation so the trainer
    converges within 4 epochs and the contract's
    ``test_loss_decreases_monotonically`` assertion fires reliably.
    """
    return TrainingConfig(
        backbone_id="densenet121",
        n_labels=2,
        n_epochs=4,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer="adamw",
        lr_schedule="constant",
        warmup_epochs=0,
        augmentations=(),
        image_size=(32, 32),
        checkpoint_dir=None,
        early_stop_patience=None,
        num_dataloader_workers=0,
    )


@pytest.mark.torch
class TestTorchFineTuneTrainerContract(TrainerPortContract):
    """Concrete contract subclass for :class:`TorchFineTuneTrainer`."""

    @pytest.fixture
    def adapter(self) -> TrainerPort:
        from harness.adapters.torch.trainer import TorchFineTuneTrainer

        return TorchFineTuneTrainer(device="cpu")

    @pytest.fixture
    def training_config(self) -> TrainingConfig:
        return _torch_finetune_tiny_config()
