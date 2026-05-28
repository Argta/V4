"""Baseline localization: time-domain cross-correlation ITD -> DOA."""

import numpy as np
from scipy.signal import butter, sosfilt

from .base import LocalizationAlgorithm, LocalizationResult


SPEED_OF_SOUND = 343.0




def _subsample_peak(x, peak_idx):
    if peak_idx <= 0 or peak_idx >= len(x) - 1:
        return float(peak_idx)
    y0, y1, y2 = x[peak_idx - 1], x[peak_idx], x[peak_idx + 1]
    denom = 2 * (2 * y1 - y0 - y2)
    if abs(denom) < 1e-12:
        return float(peak_idx)
    return float(peak_idx) + (y0 - y2) / denom


def inverse_woodworth(itd_s, head_radius=0.09, c=SPEED_OF_SOUND):
    scalar = np.isscalar(itd_s)
    itd = np.atleast_1d(np.array(itd_s, dtype=np.float64))
    max_itd = (head_radius / c) * (np.pi / 2.0 + 1.0)
    itd = np.clip(np.abs(itd), 0, max_itd * 0.999)
    lo = np.zeros_like(itd)
    hi = np.full_like(itd, np.pi / 2.0)
    for _ in range(20):
        mid = (lo + hi) / 2.0
        f_mid = (head_radius / c) * (mid + np.sin(mid)) - itd
        lo = np.where(f_mid < 0, mid, lo)
        hi = np.where(f_mid >= 0, mid, hi)
    result = (lo + hi) / 2.0
    return float(result[0]) if scalar else result


def itd_to_azimuth(itd_seconds, head_radius=0.09, stereo_frame=None,
                   fs=None, freq_range=None, lateral_override=None,
                   fb_fullband=None, prev_doa=None,
                   fb_active=False, fb_is_back=False):
    """Convert ITD to azimuth with front/back disambiguation.

    FB disambiguation priority:
    1. Active head rotation (fb_active + fb_is_back) — most reliable
    2. Spectral ILD ratio (fb_fullband provided, no active head) — static fallback
    3. Frame-to-frame continuity (prev_doa provided) — temporal smoothing
    """
    if lateral_override is not None:
        raw_az = float(lateral_override)
    else:
        lateral = inverse_woodworth(abs(itd_seconds), head_radius)
        lateral_deg = np.rad2deg(lateral)
        sign = -1 if itd_seconds < 0 else 1
        raw_az = sign * lateral_deg

    # Method 1: Active head FB (most reliable, from head rotation)
    if fb_is_back and abs(raw_az) > 15.0:
        if raw_az >= 0:
            return 180.0 - raw_az
        else:
            return -180.0 - raw_az

    # Method 2: Spectral ILD ratio for static FB disambiguation
    # Back sources: HF ILD is weaker than LF ILD (head blocks HF from behind)
    if fb_fullband is not None and fs is not None and not fb_active:
        is_behind = _spectral_fb_detect(fb_fullband, fs)
        if is_behind and abs(raw_az) > 15.0:
            if raw_az >= 0:
                return 180.0 - raw_az
            else:
                return -180.0 - raw_az

    # Method 3: Frame-to-frame continuity fallback
    if raw_az >= 0:
        flipped = 180.0 - raw_az
    else:
        flipped = -180.0 - raw_az

    if prev_doa is not None:
        d_raw = abs(((raw_az - prev_doa + 540) % 360) - 180)
        d_flip = abs(((flipped - prev_doa + 540) % 360) - 180)
        return flipped if d_flip < d_raw else raw_az
    else:
        return raw_az


def _spectral_fb_detect(stereo_frame, fs):
    """Detect if source is behind using spectral ILD ratio.

    Back sources have weaker high-frequency ILD vs low-frequency ILD
    because the head shadow blocks more HF content from behind.

    Returns True if source is likely behind the listener.
    """
    from scipy.signal import butter, sosfilt

    left = stereo_frame[:, 0]
    right = stereo_frame[:, 1]
    nyq = fs / 2

    # Pre-designed SOS filters (lazy init)
    if not hasattr(_spectral_fb_detect, '_sos_lo'):
        _spectral_fb_detect._sos_lo = butter(
            4, min(1500 / nyq, 0.95), btype='lowpass', output='sos')
        _spectral_fb_detect._sos_hi = butter(
            4, [min(2000 / nyq, 0.95), min(4000 / nyq, 0.98)],
            btype='bandpass', output='sos')

    left_lo = sosfilt(_spectral_fb_detect._sos_lo, left)
    right_lo = sosfilt(_spectral_fb_detect._sos_lo, right)
    left_hi = sosfilt(_spectral_fb_detect._sos_hi, left)
    right_hi = sosfilt(_spectral_fb_detect._sos_hi, right)

    ild_lo = 10 * np.log10((np.sum(left_lo ** 2) + 1e-10) /
                           (np.sum(right_lo ** 2) + 1e-10))
    ild_hi = 10 * np.log10((np.sum(left_hi ** 2) + 1e-10) /
                           (np.sum(right_hi ** 2) + 1e-10))

    # Back: HF ILD magnitude significantly smaller than LF ILD magnitude
    return abs(ild_hi) < abs(ild_lo) * 0.7




class XCorrITD(LocalizationAlgorithm):
    def __init__(self, fs: int, frame_duration_ms: float = 50.0,
                 frame_hop_ms: float = 25.0,
                 max_itd_ms: float = 1.0,
                 head_radius: float = 0.09,
                 freq_range: tuple = None):
        super().__init__(fs, frame_duration_ms, frame_hop_ms)
        self.max_lag = int(max_itd_ms / 1000.0 * fs)
        self.head_radius = head_radius
        self.freq_range = freq_range or (300, 3000)

    @property
    def name(self) -> str:
        return "xcorr_itd"

    def localize(self, stereo: np.ndarray) -> LocalizationResult:
        n_frames = self.n_frames(len(stereo))
        if n_frames == 0:
            return LocalizationResult(
                doa_estimated=np.array([]),
                timestamps=np.array([]),
                method=self.name,
            )

        doa = np.zeros(n_frames)
        times = np.zeros(n_frames)

        for idx, (frame, center) in enumerate(self._make_frames(stereo)):
            left = _bandpass(frame[:, 0], self.fs, self.freq_range)
            right = _bandpass(frame[:, 1], self.fs, self.freq_range)

            xcorr = np.correlate(left, right, mode="full")
            mid = len(xcorr) // 2
            s0 = mid - self.max_lag
            s1 = mid + self.max_lag + 1
            search_region = xcorr[s0:s1]
            peak_int = np.argmax(search_region)
            peak_sub = _subsample_peak(search_region, peak_int)
            peak_offset = peak_sub - self.max_lag
            itd_s = peak_offset / self.fs

            fb_frame = np.column_stack([left, right])
            fb_fullband = np.column_stack([frame[:, 0], frame[:, 1]])
            doa[idx] = itd_to_azimuth(
                itd_s, self.head_radius,
                stereo_frame=fb_frame, fs=self.fs,
                freq_range=self.freq_range,
                fb_fullband=fb_fullband,
            )
            times[idx] = center / self.fs

        return LocalizationResult(
            doa_estimated=doa,
            timestamps=times,
            method=self.name,
        )


def _bandpass(signal, fs, freq_range):
    low, high = freq_range
    nyq = fs / 2
    if low >= nyq or high <= 0:
        return signal
    low_norm = max(low / nyq, 0.01)
    high_norm = min(high / nyq, 0.99)
    if high_norm <= low_norm:
        return signal
    sos = butter(4, [low_norm, high_norm], btype="bandpass", output="sos")
    return sosfilt(sos, signal)
