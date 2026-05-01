"""Tests for the harness domain error hierarchy.

The hierarchy is rooted at ``HarnessError``. Every concrete subclass must be a
subclass of ``HarnessError`` so that callers (and contract tests) can funnel
adapter and domain failures through a single base type.
"""

from __future__ import annotations

import pytest

from harness.domain.errors import (
    AdapterError,
    ConfigError,
    ContractViolation,
    DataError,
    HarnessError,
)


class TestHarnessErrorHierarchy:
    def test_harness_error_is_an_exception(self) -> None:
        assert issubclass(HarnessError, Exception)

    @pytest.mark.parametrize(
        "subclass",
        [ConfigError, ContractViolation, DataError, AdapterError],
    )
    def test_subclass_is_harness_error(self, subclass: type[HarnessError]) -> None:
        assert issubclass(subclass, HarnessError)

    @pytest.mark.parametrize(
        "subclass",
        [ConfigError, ContractViolation, DataError, AdapterError],
    )
    def test_instance_is_harness_error(self, subclass: type[HarnessError]) -> None:
        instance = subclass("boom")
        assert isinstance(instance, HarnessError)
        assert isinstance(instance, Exception)

    def test_can_be_raised_and_caught_through_root(self) -> None:
        for subclass in (ConfigError, ContractViolation, DataError, AdapterError):
            with pytest.raises(HarnessError):
                raise subclass("boom")

    def test_subclasses_carry_message(self) -> None:
        err = ConfigError("bad seed")
        assert "bad seed" in str(err)
