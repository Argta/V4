"""Tests for binaural spatialization (HRTF)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.spatial.hrtf import (
    compute_azimuth, compute_itd, apply_itd, apply_ild,
    SPEED_OF_SOUND,
)


def test_azimuth_straight_ahead():
    """Source directly ahead: azimuth = 0 (new head-center atan2 convention)."""
    source = [2.5, 5.0, 1.5]
    head = [2.5, 2.0, 1.5]
    az = compute_azimuth(source, head)
    # atan2(0, 3) = 0 — source is straight ahead
    assert abs(az) < 0.01


def test_azimuth_right():
    """Source to the right: azimuth = pi/2."""
    source = [5.0, 2.5, 1.5]
    head = [2.5, 2.5, 1.5]
    az = compute_azimuth(source, head)
    assert 1.5 < az < 1.65  # ~ pi/2


def test_azimuth_behind():
    """Source directly behind: azimuth = pi (or -pi)."""
    source = [2.5, 0.0, 1.5]
    head = [2.5, 2.5, 1.5]
    az = compute_azimuth(source, head)
    assert abs(az) > 3.0  # ~ pi


def test_itd_max_at_90_degrees():
    """Max ITD at 90 degrees azimuth."""
    itd = compute_itd(np.pi / 2, head_radius=0.09)
    # Expected: (0.09/343) * (pi/2 + 1) ≈ 0.000675s ≈ 0.675ms
    expected = (0.09 / 343) * (np.pi / 2 + 1)
    assert abs(itd - expected) < 1e-6
    assert 0.0006 < itd < 0.0008


def test_itd_zero_at_0_degrees():
    """Zero ITD when source is straight ahead."""
    itd = compute_itd(0.0)
    assert itd == 0.0


def test_itd_zero_behind():
    """Zero ITD when source is directly behind (median plane)."""
    itd = compute_itd(np.pi)
    assert itd < 1e-12


def test_itd_back_hemisphere_folding():
    """135 deg should give same ITD as 45 deg (both 45 deg from median plane)."""
    itd_front = compute_itd(np.pi / 4)    # 45 deg front-right
    itd_back = compute_itd(3 * np.pi / 4)  # 135 deg back-right
    assert abs(itd_front - itd_back) < 1e-12


def test_apply_itd():
    """ITD should delay the far ear signal."""
    fs = 44100
    signal = np.zeros(fs)  # 1 second
    signal[0] = 1.0  # impulse at t=0

    # Far ear at 90 degrees
    delayed = apply_itd(signal, np.pi / 2, fs, is_ipsilateral=False)
    # Peak should have shifted
    peak_orig = np.argmax(signal)
    peak_delayed = np.argmax(delayed)
    assert peak_delayed > peak_orig


def test_apply_ild():
    """ILD should attenuate high frequencies for contralateral ear."""
    fs = 44100
    t = np.arange(0, 0.1, 1 / fs)
    signal = np.sin(2 * np.pi * 4000 * t)  # 4kHz tone

    # Contralateral ear at 90 degrees
    filtered = apply_ild(signal, np.pi / 2, fs, is_ipsilateral=False)
    # High frequency should be attenuated
    assert np.max(np.abs(filtered)) < np.max(np.abs(signal))


if __name__ == "__main__":
    test_azimuth_straight_ahead()
    test_azimuth_right()
    test_azimuth_behind()
    test_itd_max_at_90_degrees()
    test_itd_zero_at_0_degrees()
    test_itd_zero_behind()
    test_itd_back_hemisphere_folding()
    test_apply_itd()
    test_apply_ild()
    print("All binaural tests passed!")
