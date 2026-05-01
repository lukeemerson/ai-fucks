"""Error hierarchy for the harness domain layer.

All harness-originated exceptions inherit from :class:`HarnessError` so callers
(and contract tests) can funnel adapter and domain failures through a single
base type. Subclasses partition the failure modes by *cause*, not by *layer*:

* :class:`ConfigError` -- raised when an :class:`ExperimentConfig` (or any of
  its sub-configs) is constructed with values that violate the documented
  invariants. These are user-facing configuration mistakes.
* :class:`ContractViolation` -- raised by domain dataclasses' ``__post_init__``
  when a structural invariant is violated (e.g. probability values outside
  ``[0, 1]``, label-length mismatches, overlapping split indices). Also raised
  by contract tests when an adapter breaks its port contract.
* :class:`DataError` -- raised when *external* data (a loaded dataset, a
  decoded image, a parsed manifest) is malformed at adapter boundaries.
* :class:`AdapterError` -- raised by adapters for adapter-specific failures
  (e.g. a missing file, a third-party library error). Adapters must funnel
  every non-domain exception through this type so contract tests can assert
  the failure surface stays inside :class:`HarnessError`.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Root of the harness exception hierarchy.

    Every exception originating in ``harness/`` is an instance of this class.
    Callers wishing to catch *any* harness failure should catch this type.
    """


class ConfigError(HarnessError):
    """Raised when configuration values violate documented invariants.

    Examples: negative seeds, ``val_fraction + test_fraction >= 1``, empty
    ``label_names``, ``shrinkage`` outside ``[0, 1]``.
    """


class ContractViolation(HarnessError):
    """Raised when a structural domain invariant is violated.

    Used by ``__post_init__`` validators in :mod:`harness.domain.types` and by
    contract tests asserting that adapters honor their port's contract
    (shape/dtype/range invariants on outputs).
    """


class DataError(HarnessError):
    """Raised when external data is malformed at an adapter boundary.

    Examples: a CSV manifest with the wrong columns, a corrupt image, a label
    vocabulary that disagrees with the configured ``label_names``.
    """


class AdapterError(HarnessError):
    """Raised by adapters for adapter-specific failures.

    Adapters must wrap underlying third-party exceptions in this type (or a
    subclass) so the failure surface stays inside :class:`HarnessError`.
    """
