"""Spectral front/back detection via pinna notch frequency analysis.

Front sources produce higher pinna notch frequencies than back sources
(~8% shift), encoded by hrtf_parametric.py via fb_factor=0.92.
This module extracts those notches and computes FB likelihood.
"""

import numpy as np
from scipy.signal import butter, sosfilt, find_peaks


def detect_notch_frequencies(frame_left, frame_right, fs,
                             band=(4000, 12000)):
    """Detect the first spectral notch in each ear's high-frequency spectrum.

    Pinna notches appear as dips in the magnitude spectrum.
    The dominant notch frequency shifts with elevation and front/back.

    Args:
        frame_left: (L,) left ear signal frame
        frame_right: (L,) right ear signal frame
        fs: sample rate
        band: (low, high) frequency range to search for notches

    Returns:
        (notch_left_hz, notch_right_hz) estimated notch frequencies
    """
    nfft = 1
    while nfft < 2 * len(frame_left):
        nfft *= 2

    spec_l = np.abs(np.fft.rfft(frame_left, n=nfft))
    spec_r = np.abs(np.fft.rfft(frame_right, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Restrict to search band
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0, 0.0

    # Invert spectrum to find dips as peaks
    inv_l = -20 * np.log10(np.maximum(spec_l[mask], 1e-10))
    inv_r = -20 * np.log10(np.maximum(spec_r[mask], 1e-10))

    band_freqs = freqs[mask]

    def _find_first_notch(inv_spec):
        """Find first significant dip (peak in inverted spectrum)."""
        if len(inv_spec) < 3:
            return 0.0
        peaks, props = find_peaks(inv_spec, prominence=3.0, distance=3)
        if len(peaks) == 0:
            return 0.0
        return float(band_freqs[peaks[0]])

    notch_l = _find_first_notch(inv_l)
    notch_r = _find_first_notch(inv_r)

    return notch_l, notch_r


def spectral_fb_likelihood(left_notch_hz, right_notch_hz, fs):
    """Compute log-likelihood of front vs back given notch frequencies.

    Back sources have ~8% lower notch frequencies (fb_factor=0.92 in
    hrtf_parametric.py). This function compares observed notch positions
    to expected positions under each hypothesis.

    Args:
        left_notch_hz: detected notch frequency in left ear
        right_notch_hz: detected notch frequency in right ear
        fs: sample rate

    Returns:
        (log_p_front, log_p_back) log-likelihoods (un-normalized)
    """
    if left_notch_hz < 100 or right_notch_hz < 100:
        return 0.0, 0.0  # no reliable notch detected

    avg_notch = (left_notch_hz + right_notch_hz) / 2.0

    # Expected notch ranges (Hz) for front and back
    # Based on hrtf_parametric.py: base_notches[0] = 4200 Hz
    # front: ~4200 Hz, back: ~4200 * 0.92 ≈ 3864 Hz
    front_expected = 4200.0
    back_expected = 4200.0 * 0.92  # ≈ 3864 Hz

    sigma_notch = 500.0  # Hz, measurement uncertainty

    # Gaussian log-likelihood (un-normalized)
    log_p_front = -0.5 * ((avg_notch - front_expected) / sigma_notch) ** 2
    log_p_back = -0.5 * ((avg_notch - back_expected) / sigma_notch) ** 2

    return log_p_front, log_p_back


def compute_spectral_llr(frame_left, frame_right, fs):
    """Compute spectral LLR contribution for one frame.

    Returns:
        llr_spectral: log P(obs | H_back) - log P(obs | H_front)
    """
    notch_l, notch_r = detect_notch_frequencies(frame_left, frame_right, fs)
    log_pf, log_pb = spectral_fb_likelihood(notch_l, notch_r, fs)
    return log_pb - log_pf