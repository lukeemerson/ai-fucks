"""Contract tests for :class:`harness.ports.classifier_head.ClassifierHeadPort`.

Multi-label per-class probabilities (sigmoid, not softmax). Each adapter must
satisfy the contract regardless of the underlying learner.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from harness.adapters.fakes.classifier_head import LinearFakeClassifierHead
from harness.domain.errors import ContractViolation
from harness.ports.classifier_head import ClassifierHeadPort


def _make_xy(
    n: int, d: int, k: int, seed: int = 0
) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n, d)).astype(np.float32)
    y = (rng.random(size=(n, k)) > 0.5).astype(np.int8)
    return x, y


class ClassifierHeadPortContract:
    """Abstract contract; subclasses provide an ``adapter`` factory."""

    n_features: int = 4
    n_labels: int = 3

    @pytest.fixture
    def adapter(self) -> ClassifierHeadPort:
        raise NotImplementedError

    def test_predict_proba_before_fit_raises_contract_violation(
        self, adapter: ClassifierHeadPort
    ) -> None:
        x, _ = _make_xy(5, self.n_features, self.n_labels)
        with pytest.raises(ContractViolation):
            adapter.predict_proba(x)

    def test_fit_then_predict_proba_returns_correct_shape(
        self, adapter: ClassifierHeadPort
    ) -> None:
        x, y = _make_xy(8, self.n_features, self.n_labels, seed=1)
        adapter.fit(x, y)
        probs = adapter.predict_proba(x)
        assert probs.shape == (8, self.n_labels)

    def test_predict_proba_in_unit_interval(
        self, adapter: ClassifierHeadPort
    ) -> None:
        x, y = _make_xy(6, self.n_features, self.n_labels, seed=2)
        adapter.fit(x, y)
        probs = adapter.predict_proba(x)
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    def test_predict_proba_dtype_float32(
        self, adapter: ClassifierHeadPort
    ) -> None:
        x, y = _make_xy(6, self.n_features, self.n_labels, seed=3)
        adapter.fit(x, y)
        probs = adapter.predict_proba(x)
        assert probs.dtype == np.float32

    def test_fit_is_reproducible_for_same_data(
        self, adapter_factory: type[ClassifierHeadPort]
    ) -> None:
        x, y = _make_xy(10, self.n_features, self.n_labels, seed=4)
        a = adapter_factory()
        b = adapter_factory()
        a.fit(x, y)
        b.fit(x, y)
        pa = a.predict_proba(x)
        pb = b.predict_proba(x)
        np.testing.assert_array_equal(pa, pb)

    def test_predict_proba_is_deterministic(
        self, adapter: ClassifierHeadPort
    ) -> None:
        x, y = _make_xy(6, self.n_features, self.n_labels, seed=5)
        adapter.fit(x, y)
        first = adapter.predict_proba(x)
        second = adapter.predict_proba(x)
        np.testing.assert_array_equal(first, second)


class TestLinearFakeClassifierHeadContract(ClassifierHeadPortContract):
    @pytest.fixture
    def adapter(self) -> ClassifierHeadPort:
        return LinearFakeClassifierHead(n_labels=self.n_labels)

    @pytest.fixture
    def adapter_factory(self) -> type[ClassifierHeadPort]:
        n_labels = self.n_labels

        class _Factory:
            def __call__(self) -> ClassifierHeadPort:
                return LinearFakeClassifierHead(n_labels=n_labels)

        # Return a zero-arg callable mimicking a class.
        return _Factory()  # type: ignore[return-value]  # reason: factory protocol shim
