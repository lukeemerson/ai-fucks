"""Project-root pytest conftest.

Pytest's import logic walks up the ``__init__.py`` chain and decides that
``tests/harness/...`` test modules belong to package ``harness``. That binds
``harness`` in :data:`sys.modules` to the *test* tree (``tests/harness``),
shadowing the production ``harness`` package. This conftest pre-imports the
real package and freezes it in :data:`sys.modules` so subsequent ``import
harness`` calls resolve to the production code.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_ROOT_STR = str(_ROOT)

if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

# Force-load the production harness package by absolute file path so it wins
# over the test-tree namespace.
_SPEC = importlib.util.spec_from_file_location(
    "harness",
    str(_ROOT / "harness" / "__init__.py"),
    submodule_search_locations=[str(_ROOT / "harness")],
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not locate production harness package")
_REAL_HARNESS = importlib.util.module_from_spec(_SPEC)
sys.modules["harness"] = _REAL_HARNESS
_SPEC.loader.exec_module(_REAL_HARNESS)

# Eagerly import key submodules so they end up in sys.modules under the real
# package and cannot be replaced by the test-tree shadow during collection.
for _sub in (
    "harness.domain",
    "harness.domain.errors",
    "harness.domain.types",
    "harness.ports",
    "harness.ports.metrics",
    "harness.adapters",
    "harness.adapters.fakes",
    "harness.adapters.fakes.metrics",
    "harness.adapters.sklearn",
    "harness.adapters.sklearn.metrics",
):
    importlib.import_module(_sub)
