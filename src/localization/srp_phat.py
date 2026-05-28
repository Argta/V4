"""SRP-PHAT: Steered Response Power with Phase Transform.

Binaural version: scans candidate azimuths [-180°, 180°], computes
GCC-PHAT at each corresponding ITD, picks the direction with max power.
"""

import numpy as np

from .base import LocalizationAlgorithm, LocalizationResult
from .xcorr_itd import inverse_woodworth


class SRPPHAT(LocalizationAlgorithm):
    """Binaural SRP-PHAT DOA estimation via full-hemisphere azimuth sweep."""

    def __init__(self, fs: int, frame_duration_ms: float = 50.0,
                 frame_hop_ms: float = 25.0,
                 max_itd_ms: float = 1.0,
                 head_radius: float = 0.09,
                 freq_range: tuple = None,
                 azimuth_step_deg: float = 2.0):
        super().__init__(fs, frame_duration_ms, frame_hop_ms)
        self.max_lag = int(max_itd_ms / 1000.0 * fs)
        self.head_radius = head_radius
        self.freq_range = freq_range or (300, 3000)

        # Full hemisphere: -180° to 180°
        self.candidates_deg = np.arange(-180, 181, azimuth_step_deg)
        self.candidates_itd = np.array([
            _azimuth_to_itd_samples(az, head_radius, fs)
            for az in self.candidates_deg
        ])

    @property
    def name(self) -> str:
        return "srp_phat"

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

        nfft = 1
        while nfft < 2 * self.frame_len:
            nfft *= 2

        for idx, (frame, center) in enumerate(self._make_frames(stereo)):
            win = np.hanning(self.frame_len)
            left_w = frame[:, 0] * win
            right_w = frame[:, 1] * win

            L = np.fft.rfft(left_w, n=nfft)
            R = np.fft.rfft(right_w, n=nfft)

            X = L * np.conj(R)
            mag = np.abs(X)
            with np.errstate(divide="ignore", invalid="ignore"):
                X_phat = np.divide(X, mag, where=mag > 1e-10)
                X_phat[mag <= 1e-10] = 0

            freqs = np.fft.rfftfreq(nfft, d=1.0 / self.fs)
            mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
            X_phat[~mask] = 0

            gcc = np.fft.irfft(X_phat, n=nfft)

            srp_values = np.zeros(len(self.candidates_deg))
            for j, itd_samples in enumerate(self.candidates_itd):
                lag_idx = (nfft // 2 + itd_samples) % nfft
                srp_values[j] = gcc[lag_idx]

            best_idx = np.argmax(srp_values)
            doa[idx] = self.candidates_deg[best_idx]
            times[idx] = center / self.fs

        return LocalizationResult(
            doa_estimated=doa,
            timestamps=times,
            method=self.name,
        )


def _azimuth_to_itd_samples(azimuth_deg, head_radius, fs):
    """Convert azimuth (deg) to ITD in samples using inverse Woodworth.

    Front/back ambiguous angles produce the same |ITD|.
    SRP-PHAT resolves front/back by which candidate has higher power.
    """
    SPEED_OF_SOUND = 343.0
    # Map to lateral angle: 135° -> 45° from median plane
    lat = abs(azimuth_deg)
    if lat > 90:
        lat = 180 - lat
    lat = min(lat, 90)
    lat_rad = np.deg2rad(lat)

    # Woodworth ITD
    itd_s = (head_radius / SPEED_OF_SOUND) * (lat_rad + np.sin(lat_rad))
    itd_samples = round(itd_s * fs)
    # Sign: positive azimuth → right side → right ear leads → negative ITD in xcorr convention
    sign = 1 if azimuth_deg >= 0 else -1
    return -sign * itd_samples
