import numpy as np
from PIL import Image
from scipy.ndimage import sobel, uniform_filter

IMG_SIZE = 512  # downsample to this for speed; NIH originals are 1024x1024


def load_matrix(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _cardiothoracic_ratio(arr: np.ndarray) -> float:
    """Estimate CTR from horizontal profile of mid-thorax region."""
    h, w = arr.shape
    mid = arr[int(h * 0.35): int(h * 0.65), :]
    col_means = mid.mean(axis=0)
    threshold = np.percentile(col_means, 60)
    bright = (col_means >= threshold).astype(int)
    # find widest contiguous bright run in the central 60% of the image
    cx_lo, cx_hi = int(w * 0.20), int(w * 0.80)
    best_len = 0
    run_start = None
    for i in range(cx_lo, cx_hi + 1):
        if i < w and bright[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                best_len = max(best_len, i - run_start)
                run_start = None
    return best_len / w


def _peripheral_lucency(arr: np.ndarray) -> dict:
    """Check outer 15% strips for very dark regions (possible PTX)."""
    h, w = arr.shape
    strip = int(w * 0.15)
    left = arr[:, :strip]
    right = arr[:, -strip:]
    # low mean + low std in periphery = absent lung markings
    return {
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "left_std": float(left.std()),
        "right_std": float(right.std()),
    }


def _basal_opacification(arr: np.ndarray) -> float:
    """Mean intensity in lower 30% — elevated = effusion or consolidation."""
    h = arr.shape[0]
    basal = arr[int(h * 0.70):, :]
    return float(basal.mean())


def _bilateral_haziness(arr: np.ndarray) -> float:
    """Mean intensity of bilateral mid-zones (exclude mediastinum center)."""
    h, w = arr.shape
    mid_v = arr[int(h * 0.25): int(h * 0.65), :]
    left_zone = mid_v[:, int(w * 0.05): int(w * 0.35)]
    right_zone = mid_v[:, int(w * 0.65): int(w * 0.95)]
    return float(np.concatenate([left_zone.flatten(), right_zone.flatten()]).mean())


def _diaphragm_position(arr: np.ndarray) -> float:
    """Return relative row position of diaphragm dome (0=top, 1=bottom).
    Hyperinflated lungs push dome low (value > 0.72)."""
    h, w = arr.shape
    # diaphragm = strong horizontal edge in lower half
    lower = arr[int(h * 0.45):, int(w * 0.1): int(w * 0.9)]
    grad = np.abs(np.diff(lower.mean(axis=1)))
    peak_local = int(np.argmax(grad))
    # convert back to full image fraction
    return (int(h * 0.45) + peak_local) / h


def _focal_variance(arr: np.ndarray) -> float:
    """Max local variance across a 32x32 sliding window — flag focal opacities."""
    smoothed = uniform_filter(arr, size=32)
    local_var = uniform_filter(arr ** 2, size=32) - smoothed ** 2
    # restrict to lung field (avoid mediastinum)
    h, w = arr.shape
    lung_region = local_var[int(h * 0.15): int(h * 0.75), int(w * 0.05): int(w * 0.95)]
    return float(np.percentile(lung_region, 95))


def _horizontal_band_response(arr: np.ndarray) -> float:
    """Horizontal Sobel response in lung zones — linear atelectasis signature."""
    h, w = arr.shape
    lung = arr[int(h * 0.20): int(h * 0.80), int(w * 0.05): int(w * 0.95)]
    sx = sobel(lung, axis=0)
    return float(np.abs(sx).mean())


def extract_features(path: str) -> dict:
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
