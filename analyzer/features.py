from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.ndimage import sobel, uniform_filter

IMG_SIZE = 512  # downsample to this for speed; NIH originals are 1024x1024

# Float32 2-D pixel matrix in [0, 1]. We only ever read shape + element-wise
# numpy ops on these, so a generic float ndarray is enough; no dtype-narrow
# generics needed.
FloatArray = npt.NDArray[np.float32]


def load_matrix(path: str) -> FloatArray:
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _cardiothoracic_ratio(arr: FloatArray) -> float:
    """Estimate CTR from the widest contiguous bright run across the mid-thorax."""
    h, w = arr.shape
    mid = arr[int(h * 0.35) : int(h * 0.65), :]
    col_means = mid.mean(axis=0)
    threshold = np.percentile(col_means, 60)
    cx_lo, cx_hi = int(w * 0.20), int(w * 0.80)
    bright = col_means[cx_lo : cx_hi + 1] >= threshold
    if not bright.any():
        return 0.0
    # padding so a run touching either edge of the slice is still terminated
    padded = np.concatenate([[False], bright, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return float((ends - starts).max()) / float(w)


def _peripheral_lucency(arr: FloatArray) -> dict[str, float]:
    """Outer 15% strips — low mean + low std implies absent lung markings (possible PTX)."""
    _, w = arr.shape
    strip = int(w * 0.15)
    left = arr[:, :strip]
    right = arr[:, -strip:]
    return {
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "left_std": float(left.std()),
        "right_std": float(right.std()),
    }


def _basal_opacification(arr: FloatArray) -> float:
    """Mean intensity in lower 30% — elevated implies effusion or consolidation."""
    h = arr.shape[0]
    basal = arr[int(h * 0.70) :, :]
    return float(basal.mean())


def _bilateral_haziness(arr: FloatArray) -> float:
    """Mean intensity of bilateral mid-zones (excludes mediastinum)."""
    h, w = arr.shape
    mid_v = arr[int(h * 0.25) : int(h * 0.65), :]
    left_zone = mid_v[:, int(w * 0.05) : int(w * 0.35)]
    right_zone = mid_v[:, int(w * 0.65) : int(w * 0.95)]
    return float(np.concatenate([left_zone.flatten(), right_zone.flatten()]).mean())


def _diaphragm_position(arr: FloatArray) -> float:
    """Relative row position of diaphragm dome (0=top, 1=bottom).
    Hyperinflation pushes the dome low (>0.72)."""
    h, w = arr.shape
    lower = arr[int(h * 0.45) :, int(w * 0.1) : int(w * 0.9)]
    grad = np.abs(np.diff(lower.mean(axis=1)))
    peak_local = int(np.argmax(grad))
    return (int(h * 0.45) + peak_local) / float(h)


def _focal_variance(arr: FloatArray) -> float:
    """Max local variance across a 32x32 sliding window — flags focal opacities."""
    smoothed = uniform_filter(arr, size=32)
    local_var = uniform_filter(arr**2, size=32) - smoothed**2
    h, w = arr.shape
    lung_region = local_var[int(h * 0.15) : int(h * 0.75), int(w * 0.05) : int(w * 0.95)]
    return float(np.percentile(lung_region, 95))


def _horizontal_band_response(arr: FloatArray) -> float:
    """Horizontal Sobel response in lung zones — linear atelectasis signature."""
    h, w = arr.shape
    lung = arr[int(h * 0.20) : int(h * 0.80), int(w * 0.05) : int(w * 0.95)]
    sx = sobel(lung, axis=0)
    return float(np.abs(sx).mean())


def extract_features(path: str) -> dict[str, float]:
    arr = load_matrix(path)
    ptx = _peripheral_lucency(arr)
    return {
        "ctr": _cardiothoracic_ratio(arr),
        "ptx_left_mean": ptx["left_mean"],
        "ptx_right_mean": ptx["right_mean"],
        "ptx_left_std": ptx["left_std"],
        "ptx_right_std": ptx["right_std"],
        "basal_opacity": _basal_opacification(arr),
        "bilateral_haze": _bilateral_haziness(arr),
        "diaphragm_pos": _diaphragm_position(arr),
        "focal_variance": _focal_variance(arr),
        "horiz_band": _horizontal_band_response(arr),
    }
