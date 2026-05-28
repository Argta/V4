"""Parametric pinna model for binaural spatialization (HRTF Mode B).

Builds on the analytical spherical head model (Mode A: ITD + ILD) and adds:
- Pinna spectral notches (elevation-dependent, enables vertical localization)
- Ear canal resonance (~2.8 kHz)
- Shoulder reflection (~0.2 ms delay)
- Front/back differentiation via notch frequency shift

References:
  Batteau (1967) 鈥?pinna reflection model
  Algazi et al. (2001) 鈥?parametric spectral notches
"""

import numpy as np
from scipy.signal import iirnotch, lfilter, filtfilt

from .hrtf import (
    compute_azimuth, apply_itd, apply_ild, compute_itd, SPEED_OF_SOUND
)


def compute_elevation(source_pos, head_center):
    """Compute elevation angle from source to head.

    Args:
        source_pos: [x, y, z] source position
        head_center: [x, y, z] head center

    Returns:
        elevation in radians (0 = horizontal, +pi/2 = zenith)
    """
    rel = np.array(source_pos) - np.array(head_center)
    dist_h = np.sqrt(rel[0]**2 + rel[1]**2)
    if dist_h < 1e-6:
        return np.pi / 2 if rel[2] > 0 else -np.pi / 2
    return np.arctan2(rel[2], dist_h)


def _is_front(source_pos, head_center):
    """Determine if source is in front of the head."""
    rel = np.array(source_pos) - np.array(head_center)
    return rel[1] > 0  # positive y = front


def design_pinna_notches(elevation_rad, is_front, fs):
    """Design pinna notch filters based on elevation and front/back.

    The pinna creates spectral notches at frequencies controlled by
    the sound's elevation. Higher elevation 鈫?higher notch frequencies.
    Front vs back shifts the notch pattern.

    Args:
        elevation_rad: elevation angle in radians
        is_front: True if source is in front
        fs: sample rate

    Returns:
        list of (b, a) filter coefficient pairs for cascade IIR notches
    """
    # Normalize elevation to [-pi/2, pi/2]
    elev = np.clip(elevation_rad, -np.pi / 2, np.pi / 2)

    # Base notch frequencies (Hz) at zero elevation
    # These correspond to pinna cavity resonances
    base_notches = [4200, 6500, 8800, 11000]

    # Elevation shifts notch frequencies
    # Higher elevation 鈫?higher frequency (shorter effective cavity)
    elev_factor = 1.0 + 0.4 * np.sin(elev)  # 1.0 to 1.4

    # Front/back shift: back has ~10% lower notches (larger effective cavity)
    fb_factor = 0.92 if not is_front else 1.0

    filters = []
    for base_f in base_notches:
        f_center = base_f * elev_factor * fb_factor
        # Clamp to valid range
        if f_center > fs / 2.2:
            continue
        if f_center < 200:
            continue

        Q = 4.0 + 2.0 * abs(np.sin(elev))  # Sharper notches at higher elevations
        b, a = iirnotch(f_center, Q, fs)
        filters.append((b, a))

    return filters


def design_ear_canal_resonance(fs):
    """Design ear canal resonance filter (~2.8 kHz, Q~3).

    The ear canal acts as a quarter-wave resonator, producing a
    broad peak around 2.5-3.5 kHz with ~10-15 dB gain.

    Returns:
        (b, a) IIR peaking filter coefficients
    """
    from scipy.signal import iirpeak
    return iirpeak(2800.0, 3.0, fs)


def design_shoulder_filter(fs):
    """Design shoulder reflection filter.

    Shoulder reflection creates a delayed (~0.2 ms), attenuated copy
    that produces a comb filter effect mainly affecting frequencies
    below ~3 kHz.

    Returns:
        (b, a) FIR filter coefficients for the shoulder echo
    """
    delay_samples = int(0.0002 * fs)  # ~0.2 ms
    if delay_samples < 1:
        delay_samples = 1

    # Simple FIR: direct + attenuated delayed copy
    b = np.zeros(delay_samples + 1)
    b[0] = 1.0
    b[delay_samples] = 0.6  # Reflection coefficient
    a = np.array([1.0])

    return b, a


def process_binaural(signal_left, signal_right,
                     source_pos, left_ear_pos, right_ear_pos,
                     head_center, head_radius, fs,
                     rir_itd_present=False):
    """Apply parametric binaural spatialization (Mode B).

    Pipeline:
    1. ITD (Woodworth model) 鈥?same as Mode A
    2. ILD (Rayleigh head shadow) 鈥?same as Mode A
    3. Pinna notch filters 鈥?NEW: elevation-dependent spectral notches
    4. Ear canal resonance 鈥?NEW: broadband peak at 2.8 kHz
    5. Shoulder reflection 鈥?NEW: delayed copy producing comb filter

    Returns:
        (left_out, right_out) processed signals
    """
    # Azimuth from head center (0=ahead, +=right)
    azimuth = compute_azimuth(source_pos, head_center)
    azim_abs = abs(azimuth)
    right_is_ipsi = (azimuth >= 0)

    # Elevation and front/back
    elev = compute_elevation(source_pos, head_center)
    is_front = _is_front(source_pos, head_center)

    # 1. ITD
    if not rir_itd_present:
        left = apply_itd(signal_left, azim_abs, fs, head_radius, not right_is_ipsi)
        right = apply_itd(signal_right, azim_abs, fs, head_radius, right_is_ipsi)
    else:
        left = signal_left
        right = signal_right

    # 2. ILD
    left = apply_ild(left, azim_abs, fs, head_radius, not right_is_ipsi)
    right = apply_ild(right, azim_abs, fs, head_radius, right_is_ipsi)

    # 3. Pinna notch filters (applied to both ears)
    # lfilter: single-pass avoids excessive notch depth that
    # wipes out voice harmonics at notch frequencies
    notches = design_pinna_notches(elev, is_front, fs)
    for b, a in notches:
        left = lfilter(b, a, left).astype(np.float64)
        right = lfilter(b, a, right).astype(np.float64)

    # 4. Ear canal resonance (both ears)
    b_ec, a_ec = design_ear_canal_resonance(fs)
    left = lfilter(b_ec, a_ec, left).astype(np.float64)
    right = lfilter(b_ec, a_ec, right).astype(np.float64)

    # 5. Shoulder reflection (both ears)
    b_sh, a_sh = design_shoulder_filter(fs)
    left = lfilter(b_sh, a_sh, left)
    right = lfilter(b_sh, a_sh, right)

    # Normalize to prevent clipping
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 1.5:
        left = left / peak * 1.2
        right = right / peak * 1.2

    return left.astype(np.float64), right.astype(np.float64)
