"""Identity fake backbone: flatten ``(N, H, W, C)`` -> ``(N, H*W*C)``.

Used by tests where a real feature extractor would obscure the contract under
test. The flatten is deterministic, preserves batch order, and validates the
incoming tensor shape.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from harness.domain.errors import AdapterError


class IdentityFakeBackbone:
    """Backbone that returns the raw image tensor flattened per-sample."""

    def __init__(self, image_shape: tuple[int, int, int]) -> None:
        h, w, c = image_shape
        if h <= 0 or w <= 0 or c <= 0:
            raise AdapterError(
                f"image_shape components must be positive, got {image_shape}"
            )
        self._image_shape: tuple[int, int, int] = (h, w, c)
        self._embedding_dim: int = h * w * c

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def extract(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        if images.ndim != 4:
            raise AdapterError(
                f"expected 4-D NHWC tensor, got ndim={images.ndim} "
                f"(shape={images.shape})"
            )
        expected_hwc = self._image_shape
        actual_hwc = (
            int(images.shape[1]),
            int(images.shape[2]),
            int(images.shape[3]),
        )
        if actual_hwc != expected_hwc:
            raise AdapterError(
                f"expected image shape {expected_hwc}, got {actual_hwc}"
            )
        n = int(images.shape[0])
        return images.reshape(n, self._embedding_dim).astype(np.float32, copy=True)
