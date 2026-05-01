from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.ndimage import sobel, uniform_filter

IMG_SIZE = 512  # downsample to this for speed; NIH originals are 1024x1024
METRIC_SCHEMA_VERSION = "cxr-metrics-v3"
METRIC_KEYS = (
    "ctr",
    "ptx_left_mean",
    "ptx_right_mean",
    "ptx_left_std",
    "ptx_right_std",
    "ptx_left_edge_density",
    "ptx_right_edge_density",
    "basal_opacity",
    "bilateral_haze",
    "diaphragm_pos",
    "diaphragm_flatness",
    "focal_variance",
    "horiz_band",
    "lower_horiz_band",
)

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
    """Outer 15% strips — low mean + low std implies absent lung markings (possible PTX).
    Edge density (mean Sobel magnitude) adds a direct vascular-marking signal: PTX
    strips have almost no texture, so edge_density drops sharply."""
    _, w = arr.shape
    strip = int(w * 0.15)
    left = arr[:, :strip]
    right = arr[:, -strip:]

    def _edge_density(patch: FloatArray) -> float:
        sx = sobel(patch, axis=0)
        sy = sobel(patch, axis=1)
        return float(np.sqrt(sx**2 + sy**2).mean())

    return {
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "left_std": float(left.std()),
        "right_std": float(right.std()),
        "left_edge_density": _edge_density(left),
        "right_edge_density": _edge_density(right),
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
    y0, y1 = int(h * 0.45), int(h * 0.90)
    lower = arr[y0:y1, int(w * 0.1) : int(w * 0.9)]
    grad = np.abs(np.diff(lower.mean(axis=1)))
    peak_local = int(np.argmax(grad))
    # Exclude the bottom 10% of the frame so the detector does not lock onto
    # the image border / crop edge, which made normal films look maximally
    # hyperinflated.
    return (y0 + peak_local) / float(h)


def _diaphragm_flatness(arr: FloatArray) -> float:
    """Std of diaphragm dome position sampled at 5 column locations.

    A normal diaphragm is dome-shaped — the centre sits higher than the edges,
    producing high std. An emphysematous diaphragm is flat, producing low std.
    Complements diaphragm_pos (which only captures the overall level)."""
    h, w = arr.shape
    y0, y1 = int(h * 0.45), int(h * 0.90)
    positions: list[float] = []
    for col_frac in (0.20, 0.35, 0.50, 0.65, 0.80):
        c = int(w * col_frac)
        half = int(w * 0.07)
        col_lo = max(c - half, int(w * 0.05))
        col_hi = min(c + half, int(w * 0.95))
        strip = arr[y0:y1, col_lo:col_hi]
        grad = np.abs(np.diff(strip.mean(axis=1)))
        peak_local = int(np.argmax(grad))
        positions.append((y0 + peak_local) / float(h))
    return float(np.std(positions))


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


def _lower_zone_horiz_band(arr: FloatArray) -> float:
    """Horizontal Sobel response restricted to the lower lung zone (rows 55–80%).

    Plate and subsegmental atelectasis is predominantly bibasal. Restricting
    the band response to this zone reduces noise from upper-zone structures
    (vessels, ribs) that contaminate the full-lung horiz_band metric."""
    h, w = arr.shape
    lower = arr[int(h * 0.55) : int(h * 0.80), int(w * 0.05) : int(w * 0.95)]
    sx = sobel(lower, axis=0)
    return float(np.abs(sx).mean())


def extract_features(path: str) -> dict[str, float]:
    arr = load_matrix(path)
    ptx = _peripheral_lucency(arr)
    return {
        METRIC_KEYS[0]: _cardiothoracic_ratio(arr),
        METRIC_KEYS[1]: ptx["left_mean"],
        METRIC_KEYS[2]: ptx["right_mean"],
        METRIC_KEYS[3]: ptx["left_std"],
        METRIC_KEYS[4]: ptx["right_std"],
        METRIC_KEYS[5]: ptx["left_edge_density"],
        METRIC_KEYS[6]: ptx["right_edge_density"],
        METRIC_KEYS[7]: _basal_opacification(arr),
        METRIC_KEYS[8]: _bilateral_haziness(arr),
        METRIC_KEYS[9]: _diaphragm_position(arr),
        METRIC_KEYS[10]: _diaphragm_flatness(arr),
        METRIC_KEYS[11]: _focal_variance(arr),
        METRIC_KEYS[12]: _horizontal_band_response(arr),
        METRIC_KEYS[13]: _lower_zone_horiz_band(arr),
    }
