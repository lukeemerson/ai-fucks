"""Backbone port: image tensor -> feature vector.

Per ARCHITECTURE.md section 4.3 the backbone is the only port that touches
image tensors directly. The numpy-based shape used here is the v1 surface:
adapters accept a batched ``NDArray[float32]`` of shape ``(N, H, W, C)`` and
return a feature matrix of shape ``(N, embedding_dim)``.

Domain types are not used here because the backbone operates below the domain
boundary (raw tensors). Adapters that wrap third-party backbones are
responsible for mapping their internal shapes onto this contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class BackbonePort(Protocol):
    """Frozen feature extractor.

    Implementations are expected to be pure functions of their inputs --
    extracting the same image batch twice must yield the same feature matrix.
    """

    @property
    def embedding_dim(self) -> int:
        """Number of features each image is mapped to."""
        ...

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        """Map an ``(N, H, W, C)`` image batch to ``(N, embedding_dim)`` features.

        Adapters must raise :class:`harness.domain.errors.AdapterError` (or a
        subclass) when ``images`` does not match the expected shape or dtype.
        """
        ...
