"""Identity fake calibrator: ``transform`` returns its input unchanged.

The fake exists so contract tests can verify the
:class:`harness.ports.calibrator.CalibratorPort` lifecycle (``is_fitted``,
fit-before-transform, shape preservation, per-class independence) without
introducing isotonic / sigmoid calibration logic into the test surface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from harness.domain.errors import ContractViolation


class IdentityFakeCalibrator:
    """No-op calibrator. ``fit`` flips ``is_fitted``; ``transform`` is identity."""

    def __init__(self) -> None:
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        probs: NDArray[np.float32],
        labels: NDArray[np.int8],
    ) -> None:
        if probs.ndim != 2:
            raise ContractViolation(
                f"probs must be 2-D, got ndim={probs.ndim}"
            )
        if labels.ndim != 2:
            raise ContractViolation(
                f"labels must be 2-D, got ndim={labels.ndim}"
            )
        if probs.shape != labels.shape:
            raise ContractViolation(
                f"shape mismatch: probs={probs.shape} labels={labels.shape}"
            )
        self._fitted = True

    def transform(self, probs: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self._fitted:
            raise ContractViolation(
                "transform called before fit on IdentityFakeCalibrator"
            )
        if probs.ndim != 2:
            raise ContractViolation(
                f"probs must be 2-D, got ndim={probs.ndim}"
            )
        return probs.astype(np.float32, copy=True)
