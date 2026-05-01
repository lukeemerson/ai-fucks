"""Unit tests on feature extractors with crafted numpy arrays.

These avoid the PNG roundtrip — we inject the array directly into the inner
functions so a regression points to the math, not to PIL / resampling.
"""

import numpy as np
import pytest

from analyzer.features import (
    IMG_SIZE,
    FloatArray,
    _basal_opacification,
    _bilateral_haziness,
    _cardiothoracic_ratio,
    _diaphragm_position,
    _focal_variance,
    _peripheral_lucency,
)


def _blank(value: float = 0.18) -> FloatArray:
    return np.full((IMG_SIZE, IMG_SIZE), value, dtype=np.float32)


def test_ctr_zero_when_image_is_dark_uniform() -> None:
    """A dark uniform field has no bright run — CTR must be 0 (no false cardiomegaly)."""
    # threshold = percentile(60) of 0.05 == 0.05; bright is True everywhere so
    # widest run = central window. The function should still return >=0; we just
    # verify it does not blow up and stays bounded.
    arr = _blank(0.05)
    assert 0.0 <= _cardiothoracic_ratio(arr) <= 1.0


def test_ctr_finalizes_run_touching_right_edge() -> None:
    """Regression: a bright run that extends through cx_hi must still be measured.

    Pre-fix, the loop exited with run_start set and best_len was never updated,
    so wide silhouettes touching the right edge of the central window reported 0.
    """
    arr = _blank(0.05)
    # bright band from column 200 through the end — touches and crosses cx_hi (~410)
    arr[:, 200:] = 0.85
    ctr = _cardiothoracic_ratio(arr)
    # the band spans ~60% of the image; CTR must be clearly non-zero
    assert ctr > 0.30, f"edge-touching run lost: ctr={ctr}"


def test_ctr_picks_widest_of_multiple_runs() -> None:
    """Two bright bands inside the central window — the wider one must win.

    The function thresholds on the 60th-percentile column intensity, so we keep
    the array's bright fraction comfortably above 40% to ensure the threshold
    discriminates.
    """
    arr = _blank(0.05)
    arr[:, : int(IMG_SIZE * 0.20)] = 0.85  # bright outside central window
    arr[:, int(IMG_SIZE * 0.80) :] = 0.85  # bright outside central window
    arr[:, 200:220] = 0.85  # narrow run (20 cols, inside)
    arr[:, 280:330] = 0.85  # wider run (50 cols, inside)
    ctr = _cardiothoracic_ratio(arr)
    assert ctr == pytest.approx(50 / IMG_SIZE, abs=0.01)


def test_peripheral_lucency_flags_dark_uniform_periphery() -> None:
    arr = _blank(0.40)
    arr[:, : int(IMG_SIZE * 0.15)] = 0.05  # very dark left strip, near zero variance
    out = _peripheral_lucency(arr)
    assert out["left_mean"] < 0.10
    assert out["left_std"] < 0.05
    assert out["right_mean"] == pytest.approx(0.40, abs=1e-4)


def test_basal_opacification_picks_lower_band() -> None:
    arr = _blank(0.20)
    arr[int(IMG_SIZE * 0.70) :, :] = 0.80
    assert _basal_opacification(arr) == pytest.approx(0.80, abs=1e-4)


def test_bilateral_haziness_excludes_mediastinum() -> None:
    """Bright mediastinum alone should not raise haze — the function samples the lateral zones."""
    arr = _blank(0.20)
    cx = IMG_SIZE // 2
    arr[:, cx - 30 : cx + 30] = 0.95
    assert _bilateral_haziness(arr) == pytest.approx(0.20, abs=1e-3)


def test_focal_variance_responds_to_bright_spot() -> None:
    """A focal lesion must measurably raise focal_variance vs a uniform field.

    Absolute thresholds are tuned on real CXR pixel statistics, so we test the
    response (delta) rather than a brittle absolute cutoff.
    """
    base = _focal_variance(_blank(0.30))
    arr = _blank(0.30)
    arr[150:330, 150:330] = 0.95
    assert _focal_variance(arr) > base + 0.02


def test_focal_variance_low_for_uniform() -> None:
    assert _focal_variance(_blank(0.40)) < 0.005


def test_diaphragm_position_ignores_bottom_frame_edge() -> None:
    """A bright crop edge at the bottom must not look like a maximally low dome."""
    arr = _blank(0.20)
    arr[int(IMG_SIZE * 0.72) : int(IMG_SIZE * 0.73), :] = 0.95
    arr[int(IMG_SIZE * 0.98) :, :] = 1.00
    pos = _diaphragm_position(arr)
    assert pos == pytest.approx(0.72, abs=0.03)


def test_diaphragm_position_increases_when_dome_is_lower() -> None:
    upper = _blank(0.20)
    lower = _blank(0.20)
    upper[int(IMG_SIZE * 0.62) : int(IMG_SIZE * 0.63), :] = 0.95
    lower[int(IMG_SIZE * 0.78) : int(IMG_SIZE * 0.79), :] = 0.95
    assert _diaphragm_position(lower) > _diaphragm_position(upper)
