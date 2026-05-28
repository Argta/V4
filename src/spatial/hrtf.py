"""Spherical head model for binaural spatialization.

Provides:
- ITD (Interaural Time Difference) via Woodworth formula
- ILD (Interaural Level Difference) via Rayleigh head shadow model

Reference:
  Woodworth & Schlosberg (1954) - Experimental Psychology
  Rayleigh (1907) - On the diffraction of sound by a rigid sphere
"""

import numpy as np
from scipy.signal import butter, filtfilt

SPEED_OF_SOUND = 343.0


# --- Azimuth computation ---



def compute_azimuth(source_pos, head_center):
    """Compute horizontal azimuth from head center.

    0 = straight ahead (+y), positive = right, negative = left.
    Uses atan2(x, y) for correct signed angle.
    """
    src = np.array(source_pos)
    head = np.array(head_center)
    rel = src - head
    # atan2(x, y): x=interaural (right+), y=forward
    return np.arctan2(rel[0], rel[1])


# --- ITD: Woodworth spherical head model ---

def compute_itd(azimuth_rad: float, head_radius: float = 0.09,
                c: float = SPEED_OF_SOUND) -> float:
    """Compute Interaural Time Difference in seconds.

    Uses the Woodworth formula: ITD = (a/c) * (theta + sin(theta))
    where a = head radius, theta = |azimuth|.

    Args:
        azimuth_rad: Source azimuth in radians (0 = ahead, pi/2 = far right)
        head_radius: Head radius in meters (default 0.09 = adult human)
        c: Speed of sound in m/s

    Returns:
        ITD in seconds (positive value)
    """
    theta = abs(azimuth_rad)
    # Fold back hemisphere: 135° -> 45° from median plane
    if theta > np.pi / 2:
        theta = np.pi - theta
    return (head_radius / c) * (theta + np.sin(theta))


def apply_itd(signal: np.ndarray, itd_or_azimuth: float, fs: int,
              head_radius: float = 0.09, is_ipsilateral: bool = True,
              direct_itd: bool = False, blend: float = None) -> np.ndarray:
    """Apply ITD via frequency-domain phase shift.

    Args:
        itd_or_azimuth: ITD in seconds (if direct_itd=True) or azimuth in radians
        direct_itd: if True, first arg is ITD seconds (not azimuth)
        blend: if provided, applies blend*ITD delay (overrides is_ipsilateral)
    """
    if blend is not None:
        if blend < 1e-6:
            return signal
    elif is_ipsilateral:
        return signal

    if direct_itd:
        itd_s = itd_or_azimuth
    else:
        itd_s = compute_itd(itd_or_azimuth, head_radius)

    if blend is not None:
        itd_s *= blend

    if itd_s < 1e-6:
        return signal

    n = len(signal)
    nfft = 1
    while nfft < n:
        nfft *= 2
    X = np.fft.rfft(signal, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    phase_shift = -2j * np.pi * freqs * itd_s  # negative = delay
    X *= np.exp(phase_shift)
    return np.fft.irfft(X, n=nfft)[:n].astype(np.float64)


# --- ILD: Rayleigh spherical head shadow ---

def design_ild_filter(azimuth_rad: float, fs: int,
                      head_radius: float = 0.09,
                      c: float = SPEED_OF_SOUND):
    """Design a lowpass filter modeling head shadow for the contralateral ear.

    Based on Rayleigh's rigid sphere diffraction model:
    - Low frequencies diffract around the head (little attenuation)
    - High frequencies are shadowed (significant attenuation)
    - Cutoff frequency fc = c / (2 * pi * a * sin(theta))

    Args:
        azimuth_rad: Source azimuth (magnitude)
        fs: Sample rate
        head_radius: Head radius in meters
        c: Speed of sound

    Returns:
        (b, a) filter coefficients for a 1st-order lowpass filter
    """
    theta = abs(azimuth_rad)
    # Fold back hemisphere: 135° -> 45° from median plane
    if theta > np.pi / 2:
        theta = np.pi - theta
    if theta < 0.05:
        return (np.array([1.0]), np.array([1.0]))  # No attenuation

    sin_theta = np.sin(theta)
    if sin_theta < 1e-6:
        return (np.array([1.0]), np.array([1.0]))

    fc = c / (2 * np.pi * head_radius * sin_theta)
    fc = np.clip(fc, 300, 6000)  # Bounds for realistic ILD

    b, a = butter(2, fc / (fs / 2), btype="lowpass")
    return b, a


def apply_ild(signal: np.ndarray, azimuth_rad: float, fs: int,
              head_radius: float = 0.09,
              is_ipsilateral: bool = True) -> np.ndarray:
    """Apply frequency-dependent ILD via Rayleigh head shadow filter.

    The ipsilateral ear receives the signal unchanged.
    The contralateral ear receives a lowpass-filtered version
    (high frequencies attenuated by head shadow).

    Args:
        signal: 1-D input signal
        azimuth_rad: Source azimuth from this ear's perspective
        fs: Sample rate
        head_radius: Head radius in meters
        is_ipsilateral: True if this is the ear on the source side

    Returns:
        Filtered signal (same length)
    """
    if is_ipsilateral:
        return signal
    b, a = design_ild_filter(azimuth_rad, fs, head_radius)
    if len(b) == 1 and b[0] == 1.0:
        return signal  # No attenuation needed (near median plane)
    return filtfilt(b, a, signal.astype(np.float64)).astype(np.float64)
