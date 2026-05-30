"""Dual-band binaural localization: ITD (300-1500Hz) + ILD (2000-4000Hz).

ITD is most reliable at low-mid frequencies where wavelength > head diameter
and phase wrapping is rare. ILD is strongest at high frequencies where head
shadowing is most pronounced. Separate bands + complementary fusion.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from .base import LocalizationAlgorithm, LocalizationResult
from .xcorr_itd import itd_to_azimuth, _subsample_peak, inverse_woodworth


class GCCPHAT(LocalizationAlgorithm):
    """Dual-band DOA estimation: ITD from LF band, ILD from HF band."""

    def __init__(self, fs: int, frame_duration_ms: float = 50.0,
                 frame_hop_ms: float = 25.0,
                 max_itd_ms: float = 1.0,
                 head_radius: float = 0.09,
                 freq_range: tuple = None,
                 head_yaw_speed: float = 0.0,
                 active_head: bool = False):
        super().__init__(fs, frame_duration_ms, frame_hop_ms)
        self.max_lag = int(max_itd_ms / 1000.0 * fs)
        self.head_radius = head_radius
        self.itd_band = (300, 3000)
        self.ild_band = (2000, 4000)
        self.head_yaw_speed = head_yaw_speed
        self.active_head = active_head

        # Pre-design filters
        nyq = fs / 2
        self.sos_itd = butter(4, [self.itd_band[0]/nyq, min(self.itd_band[1]/nyq, 0.98)],
                              btype='bandpass', output='sos')
        self.sos_ild = butter(2, [min(self.ild_band[0]/nyq, 0.95),
                                  min(self.ild_band[1]/nyq, 0.98)],
                              btype='bandpass', output='sos')
        self.sos_ild_lf = butter(2, [300/nyq, min(800/nyq, 0.98)],
                                 btype='bandpass', output='sos')

    @property
    def name(self) -> str:
        return "gcc_phat"

    def localize(self, stereo: np.ndarray) -> LocalizationResult:
        n_frames = self.n_frames(len(stereo))
        if n_frames == 0:
            return LocalizationResult(doa_estimated=np.array([]),
                                      timestamps=np.array([]), method=self.name)

        doa = np.zeros(n_frames); times = np.zeros(n_frames)
        diag_itd = np.zeros(n_frames); diag_lat = np.zeros(n_frames)
        diag_phase = np.zeros(n_frames); diag_freq = np.zeros(n_frames)
        doa_itd = np.zeros(n_frames); doa_ild = np.zeros(n_frames)

        detail_frame = n_frames // 2; detail_data = {}
        last_valid_itd = 0.0

        nfft = 1
        while nfft < 2 * self.frame_len:
            nfft *= 2

        # Active head analysis state
        fb_determined = not self.active_head  # skip if disabled
        is_back = False
        ild_early = []; itd_rotate = []

        for idx, (frame, center) in enumerate(self._make_frames(stereo)):
            win = np.hanning(self.frame_len)
            frame_start = center - self.frame_len // 2

            # ── Low band (300-1500 Hz): ITD extraction ──
            l_itd = sosfilt(self.sos_itd, frame[:, 0] * win)
            r_itd = sosfilt(self.sos_itd, frame[:, 1] * win)

            L = np.fft.rfft(l_itd, n=nfft)
            R = np.fft.rfft(r_itd, n=nfft)
            X = L * np.conj(R)
            freqs = np.fft.rfftfreq(nfft, d=1.0 / self.fs)

            # Soft-PHAT: notch-immune phase + magnitude weighting
            mag = np.abs(X)
            soft_mag = np.sqrt(mag)
            with np.errstate(divide="ignore", invalid="ignore"):
                X_soft = np.divide(X, mag, where=mag > 1e-10) * soft_mag
                X_soft[mag <= 1e-10] = 0
            # Zero outside ITD band
            fmask_itd = (freqs >= self.itd_band[0]) & (freqs <= self.itd_band[1])
            X_soft[~fmask_itd] = 0

            # Narrowband detection: pure tone → phase method
            itd_bins = np.where(fmask_itd)[0]
            itd_mags = np.abs(X[itd_bins])
            max_ratio = np.max(itd_mags) / (np.sum(itd_mags) + 1e-10)
            # Threshold: one bin > 60% of total energy = narrowband (pure tone)
            is_narrow = (max_ratio > 0.6)

            if is_narrow:
                dom_idx = np.argmax(itd_mags)
                dom_f = freqs[itd_bins][dom_idx]
                dom_phase = np.angle(X[itd_bins][dom_idx])
                itd_s = dom_phase / (2.0 * np.pi * dom_f)
                peak_val = float(itd_mags[dom_idx])
            else:
                gcc = np.fft.irfft(X_soft, n=nfft)
                search = np.concatenate([gcc[-self.max_lag:], gcc[:self.max_lag+1]])
                peak_int = np.argmax(search)
                peak_sub = _subsample_peak(search, peak_int)
                itd_s = (peak_sub - self.max_lag) / self.fs
                peak_val = float(search[peak_int])

            # Energy gate
            frame_rms = np.sqrt(np.mean(l_itd**2 + r_itd**2))
            if frame_rms < 1e-4:
                itd_s = last_valid_itd
            else:
                last_valid_itd = itd_s

            from .xcorr_itd import inverse_woodworth
            lateral_itd = np.rad2deg(inverse_woodworth(abs(itd_s), self.head_radius))

            # ── ILD from processed signal ──
            l_hf = sosfilt(self.sos_ild, frame[:, 0] * win)
            r_hf = sosfilt(self.sos_ild, frame[:, 1] * win)
            l_lf = sosfilt(self.sos_ild_lf, frame[:, 0] * win)
            r_lf = sosfilt(self.sos_ild_lf, frame[:, 1] * win)
            ild_hf = 10 * np.log10((np.sum(l_hf**2) + 1e-10) / (np.sum(r_hf**2) + 1e-10))
            ild_lf = 10 * np.log10((np.sum(l_lf**2) + 1e-10) / (np.sum(r_lf**2) + 1e-10))
            # ILD determines left/right (ITD sign unreliable near 0° OLA crossing)
            sign = -1 if ild_hf > 0 else 1
            # Ratio LF/HF monotonic near 90°: HF saturates early, LF rises late
            ild_ratio = np.clip(np.abs(ild_lf) / (np.abs(ild_hf) + 1e-10), 0, 1)
            theta_ild = 60.0 + ild_ratio * 30.0

            # ── Fusion: ITD reliable <70°, ILD takes over >70° ──
            if lateral_itd > 70.0 and theta_ild < lateral_itd:
                lateral_fused = theta_ild
            else:
                lateral_fused = lateral_itd

            doa_fused = sign * lateral_fused
            # ITD-band frame for DOA; full-band frame for pinna front/back detection
            fb_frame = np.column_stack([l_itd, r_itd])
            fb_fullband = np.column_stack([frame[:, 0] * win, frame[:, 1] * win])

            # Active head: determine FB during rotation phase
            if not fb_determined:
                if idx < 4:                    # 0-0.1s: ILD collection
                    ild_early.append(float(ild_hf))
                elif 4 <= idx < 20:            # 0.1-0.5s: ITD slope (60deg/s)
                    itd_rotate.append(float(itd_s))
                if idx == 20 and len(itd_rotate) > 8:
                    source_left = np.mean(ild_early) > 0
                    t_rot = np.arange(len(itd_rotate)) * self.frame_hop / self.fs
                    slope = np.polyfit(t_rot, itd_rotate, 1)[0]
                    # Head rotates RIGHT. Source LEFT = away from source → flip sign
                    if source_left:
                        slope = -slope
                    # negative slope = ITD decreasing = source approaches midline = FRONT
                    is_back = slope > 0
                    fb_determined = True

            prev = None if idx == 0 else doa[idx - 1]
            doa[idx] = itd_to_azimuth(itd_s, self.head_radius,
                                       stereo_frame=fb_frame, fs=self.fs,
                                       freq_range=self.itd_band,
                                       lateral_override=doa_fused,
                                       fb_fullband=fb_fullband,
                                       prev_doa=prev,
                                       fb_active=fb_determined,
                                       fb_is_back=is_back)
            doa_itd[idx] = itd_to_azimuth(itd_s, self.head_radius,
                                           stereo_frame=fb_frame, fs=self.fs,
                                           freq_range=self.itd_band,
                                           fb_fullband=fb_fullband,
                                           prev_doa=prev)
            doa_ild[idx] = itd_to_azimuth(itd_s, self.head_radius,
                                           stereo_frame=fb_frame, fs=self.fs,
                                           freq_range=self.itd_band,
                                           lateral_override=sign * theta_ild,
                                           fb_fullband=fb_fullband,
                                           prev_doa=prev)
            times[idx] = center / self.fs
            diag_itd[idx] = itd_s
            diag_lat[idx] = sign * lateral_fused
            diag_phase[idx] = float(ild_hf)
            diag_freq[idx] = float(ild_ratio)

            # Detail capture
            if idx == detail_frame and not is_narrow:
                gcc_detail = gcc
                detail_data = {
                    'left': l_itd, 'right': r_itd,
                    'freqs': freqs.copy(),
                    'spec_l': np.abs(L), 'spec_r': np.abs(R),
                    'xcorr': gcc_detail,
                    'xcorr_lag': (lambda lags: np.where(
                        lags > nfft // 2, lags - nfft, lags)
                    )(np.arange(nfft)) / self.fs * 1e6,
                }

        # Median filter cleanup
        from scipy.signal import medfilt
        diag_itd = medfilt(diag_itd, kernel_size=5)
        doa = medfilt(doa, kernel_size=5)
        diag_lat = medfilt(diag_lat, kernel_size=5)
        doa_itd = medfilt(doa_itd, kernel_size=5)
        doa_ild = medfilt(doa_ild, kernel_size=5)

        result = LocalizationResult(
            doa_estimated=doa, timestamps=times, method=self.name,
            itd_per_frame=diag_itd, lateral_angle=diag_lat,
            phase_mean=diag_phase, freq_mean=diag_freq,
            doa_itd_only=doa_itd, doa_ild_only=doa_ild,
        )
        if detail_data:
            result.diag_frame_left = detail_data['left']
            result.diag_frame_right = detail_data['right']
            result.diag_freqs = detail_data['freqs']
            result.diag_spec_left = detail_data['spec_l']
            result.diag_spec_right = detail_data['spec_r']
            result.diag_xphase = detail_data['xcorr']
            result.diag_xweight = detail_data['xcorr_lag']
        return result


    # ---- Streaming API (Phase 1) ----

    def reset(self):
        """Clear internal state for a new streaming session."""
        super().reset()
        self._last_valid_itd = 0.0
        self._fb_determined = not self.active_head
        self._is_back = False
        self._ild_early = []
        self._itd_rotate = []
        self._t_rotate = []
        self._nfft = 1
        while self._nfft < 2 * self.frame_len:
            self._nfft *= 2
        self._accum_time = 0.0

    def process_frame(self, frame, yaw_head=0.0):
        """Process a single stereo frame. Returns (doa_world_deg, confidence).

        Args:
            frame: (frame_len, 2) stereo samples
            yaw_head: current head yaw (deg), 0=faces forward

        Returns:
            (doa_world_deg, confidence) tuple
        """
        # -- NFFT (lazy) --
        if not hasattr(self, "_nfft"):
            self._nfft = 1
            while self._nfft < 2 * self.frame_len:
                self._nfft *= 2
        nfft = self._nfft

        win = np.hanning(self.frame_len)

        # -- ITD extraction via helper --
        itd_s, gcc, freqs, L, R = _gcc_phat_single_frame(
            frame, win, self.max_lag, nfft,
            self.sos_itd, self.itd_band, self.fs)

        # Energy gate
        l_itd = sosfilt(self.sos_itd, frame[:, 0] * win)
        r_itd = sosfilt(self.sos_itd, frame[:, 1] * win)
        frame_rms = np.sqrt(np.mean(l_itd ** 2 + r_itd ** 2))
        if frame_rms < 1e-4:
            itd_s = getattr(self, "_last_valid_itd", 0.0)
        else:
            self._last_valid_itd = itd_s

        # -- ILD from dual bands --
        lateral_itd = np.rad2deg(inverse_woodworth(abs(itd_s), self.head_radius))

        l_hf = sosfilt(self.sos_ild, frame[:, 0] * win)
        r_hf = sosfilt(self.sos_ild, frame[:, 1] * win)
        l_lf = sosfilt(self.sos_ild_lf, frame[:, 0] * win)
        r_lf = sosfilt(self.sos_ild_lf, frame[:, 1] * win)
        ild_hf = 10 * np.log10((np.sum(l_hf ** 2) + 1e-10) / (np.sum(r_hf ** 2) + 1e-10))
        ild_lf = 10 * np.log10((np.sum(l_lf ** 2) + 1e-10) / (np.sum(r_lf ** 2) + 1e-10))
        sign = -1 if ild_hf > 0 else 1
        ild_ratio = np.clip(np.abs(ild_lf) / (np.abs(ild_hf) + 1e-10), 0, 1)
        theta_ild = 60.0 + ild_ratio * 30.0

        # Fusion
        if lateral_itd > 70.0 and theta_ild < lateral_itd:
            lateral_fused = theta_ild
        else:
            lateral_fused = lateral_itd
        doa_fused = sign * lateral_fused

        # -- Active-head front/back --
        fb_frame = np.column_stack([l_itd, r_itd])
        fb_fullband = np.column_stack([frame[:, 0] * win, frame[:, 1] * win])

        if self.active_head and not self._fb_determined:
            idx = self._frame_idx
            if idx < 4:
                self._ild_early.append(ild_hf)
            elif 4 <= idx < 20:
                t_rel = (idx - 4) * self.frame_hop / self.fs
                self._t_rotate.append(t_rel)
                self._itd_rotate.append(abs(itd_s))
            if idx == 19:
                source_left = float(np.mean(self._ild_early)) > 0
                if len(self._itd_rotate) > 2:
                    slope = np.polyfit(self._t_rotate, self._itd_rotate, 1)[0]
                    if source_left:
                        slope = -slope
                    self._is_back = slope > 0
                self._fb_determined = True

        prev = self._prev_doa
        doa_head = itd_to_azimuth(
            itd_s, self.head_radius,
            stereo_frame=fb_frame, fs=self.fs,
            freq_range=self.itd_band,
            lateral_override=doa_fused,
            fb_fullband=fb_fullband,
            prev_doa=prev,
            fb_active=self._fb_determined,
            fb_is_back=self._is_back)

        # World-frame
        doa_world = doa_head + yaw_head
        doa_world = ((doa_world + 180) % 360) - 180

        # Confidence based on GCC peak height
        search = np.concatenate([gcc[-self.max_lag:], gcc[:self.max_lag + 1]])
        peak_val = float(np.max(search))
        conf = min(1.0, max(0.0, peak_val / (frame_rms * 5 + 1e-10)))

        # -- Update state --
        self._prev_doa = doa_head
        self._frame_idx += 1
        self._accum_time += self.frame_hop / self.fs

        return doa_world, conf



def _gcc_phat_single_frame(frame, win, max_lag, nfft, sos_itd, freq_range, fs):
    """Extract ITD from a single frame via GCC-PHAT.

    Standalone function usable by LLRLocator and other algorithms.

    Args:
        frame: (L, 2) stereo frame
        win: (L,) window function
        max_lag: max lag in samples
        nfft: FFT size
        sos_itd: SOS bandpass filter for ITD band
        freq_range: (low, high) frequency range for masking
        fs: sample rate

    Returns:
        (itd_s, gcc, freqs, L, R) where itd_s is ITD in seconds
    """
    l_itd = sosfilt(sos_itd, frame[:, 0] * win)
    r_itd = sosfilt(sos_itd, frame[:, 1] * win)

    L = np.fft.rfft(l_itd, n=nfft)
    R = np.fft.rfft(r_itd, n=nfft)
    X = L * np.conj(R)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    fmask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    mag = np.abs(X)
    soft_mag = np.sqrt(mag)
    with np.errstate(divide="ignore", invalid="ignore"):
        X_soft = np.divide(X, mag, where=mag > 1e-10) * soft_mag
        X_soft[mag <= 1e-10] = 0
    X_soft[~fmask] = 0

    gcc = np.fft.irfft(X_soft, n=nfft)
    search = np.concatenate([gcc[-max_lag:], gcc[:max_lag + 1]])
    peak_int = np.argmax(search)
    peak_sub = _subsample_peak(search, peak_int)
    itd_s = (peak_sub - max_lag) / fs

    return itd_s, gcc, freqs, L, R