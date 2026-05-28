
# ═══════════════════════════════════════════════════════════════════════════
# DEPRECATED since v4.0 — replaced by LLRLocator (llr_locator.py).
# Retained as a baseline benchmark. Use method="llr" instead.
# ═══════════════════════════════════════════════════════════════════════════
"""Active binaural localizer — pure stereo (N,2) input.

Phase 1 (0-0.1s):  ILD determines left/right (head still)
Phase 2 (0.1-0.35s): Head turns RIGHT 15deg at 60deg/s, ITD slope → front/back
Phase 3: GCC-PHAT per-frame DOA with FB correction from Phase 2 result

No deadband, no lock, no EMA, no ground truth. Only stereo.
FB correction suppressed near median plane (|raw_az| < 15deg).
"""

import numpy as np
from scipy.signal import butter, sosfilt

from .base import LocalizationAlgorithm, LocalizationResult
from .xcorr_itd import itd_to_azimuth, _subsample_peak


class ActiveLocator(LocalizationAlgorithm):
    """Active localization from stereo only — Phase 1 ILD + Phase 2 ITD-slope + GCC-PHAT."""

    def __init__(self, fs: int, frame_duration_ms: float = 50.0,
                 frame_hop_ms: float = 25.0,
                 max_itd_ms: float = 1.0,
                 head_radius: float = 0.09,
                 freq_range: tuple = None,
                 verbose: bool = True):
        super().__init__(fs, frame_duration_ms, frame_hop_ms)
        self.verbose = verbose
        self.max_lag = int(max_itd_ms / 1000.0 * fs)
        self.head_radius = head_radius
        self.itd_band = freq_range or (300, 3000)
        self.ild_band = (2000, 4000)
        nyq = fs / 2
        self.sos_itd = butter(4, [self.itd_band[0]/nyq,
                              min(self.itd_band[1]/nyq, 0.98)],
                              btype='bandpass', output='sos')
        self.sos_ild = butter(2, [min(self.ild_band[0]/nyq, 0.95),
                                  min(self.ild_band[1]/nyq, 0.98)],
                              btype='bandpass', output='sos')

        # Generator schedule (frame-aligned, at 25ms hop):
        #   Phase 1:  frames 0-3   (0-0.1s, still)
        #   Phase 2:  frames 4-13  (0.1-0.35s, right 15deg at 60deg/s)
        #   Flash:    frames 14-25 (0.35-0.6s, ~120deg at 480deg/s — only for back)
        #   Settling: frames 26-27 (0.6-0.65s, still — only after flash)
        #   Tracking: frame  14+  (front) / frame 26+ (back)
        self.p1_frames = 4
        self.p2_start = 4
        self.p2_frames = 10               # 0.25s at 25ms hop
        self.detect_frame = self.p2_start + self.p2_frames  # 14 (=0.35s)
        self.track_start_back = 26        # after flash + settling

    @property
    def name(self) -> str:
        return "active_locator"

    def localize(self, stereo: np.ndarray) -> LocalizationResult:
        n_frames = self.n_frames(len(stereo))
        if n_frames == 0:
            return LocalizationResult(doa_estimated=np.array([]),
                                      timestamps=np.array([]), method=self.name)

        # === Phase 1: ILD left/right ===
        source_left = self._phase1_detect_side(stereo)
        if self.verbose:
            print(f'  [Phase1] source_left={source_left}')

        # === Phase 2: ITD slope front/back ===
        is_back = self._phase2_detect_fb(stereo, source_left, self.p2_start)
        if self.verbose:
            print(f'  [Phase2] is_back={is_back}')

        # For back sources, flash moves the source to front hemisphere,
        # so FB correction is only applied during the pre-flash tracking gap.
        # After tracking starts, we suppress FB (source is effectively front).
        # The FB deadband in itd_to_azimuth also prevents flips near the midline.
        fb_is_back = is_back
        detect_frame = self.track_start_back if is_back else self.detect_frame

        # === Phase 3: Full-frame GCC-PHAT DOA ===
        doa = np.zeros(n_frames); times = np.zeros(n_frames)
        diag_itd = np.zeros(n_frames); diag_phase = np.zeros(n_frames)

        nfft = 1
        while nfft < 2 * self.frame_len:
            nfft *= 2

        for idx, (frame, center) in enumerate(self._make_frames(stereo)):
            win = np.hanning(self.frame_len)
            l_itd = sosfilt(self.sos_itd, frame[:, 0] * win)
            r_itd = sosfilt(self.sos_itd, frame[:, 1] * win)
            X = np.fft.rfft(l_itd, n=nfft) * np.conj(np.fft.rfft(r_itd, n=nfft))
            freqs = np.fft.rfftfreq(nfft, d=1.0 / self.fs)
            fmask = (freqs >= self.itd_band[0]) & (freqs <= self.itd_band[1])

            mag = np.abs(X)
            soft_mag = np.sqrt(mag)
            with np.errstate(divide="ignore", invalid="ignore"):
                X_soft = np.divide(X, mag, where=mag > 1e-10) * soft_mag
                X_soft[mag <= 1e-10] = 0
            X_soft[~fmask] = 0
            gcc = np.fft.irfft(X_soft, n=nfft)
            search = np.concatenate([gcc[-self.max_lag:], gcc[:self.max_lag+1]])
            peak_int = np.argmax(search)
            peak_sub = _subsample_peak(search, peak_int)
            itd_s = (peak_sub - self.max_lag) / self.fs

            fb_frame = np.column_stack([l_itd, r_itd])
            # After flash, back sources are in front hemisphere → suppress FB
            fb_on = fb_is_back if idx >= detect_frame else False
            doa[idx] = itd_to_azimuth(itd_s, self.head_radius,
                                       stereo_frame=fb_frame, fs=self.fs,
                                       freq_range=self.itd_band,
                                       fb_is_back=fb_on)
            times[idx] = center / self.fs
            diag_itd[idx] = itd_s

        # ILD energy for diagnostic
        for idx, (frame, _) in enumerate(self._make_frames(stereo)):
            if idx >= n_frames:
                break
            win = np.hanning(self.frame_len)
            l_hf = sosfilt(self.sos_ild, frame[:, 0] * win)
            r_hf = sosfilt(self.sos_ild, frame[:, 1] * win)
            ild_db = 10 * np.log10((np.sum(l_hf**2) + 1e-10) / (np.sum(r_hf**2) + 1e-10))
            diag_phase[idx] = float(ild_db)

        return LocalizationResult(
            doa_estimated=doa, timestamps=times, method=self.name,
            itd_per_frame=diag_itd, phase_mean=diag_phase,
        )

    def _phase1_detect_side(self, stereo: np.ndarray, start_frame: int = 0) -> bool:
        """ILD-based left/right from given start frame. True = source LEFT."""
        n_frames_min = min(self.n_frames(len(stereo)) - start_frame, self.p1_frames)
        if n_frames_min < 2:
            return False
        ild_values = []
        for idx, (frame, _) in enumerate(self._make_frames(stereo)):
            if idx < start_frame:
                continue
            if idx >= start_frame + n_frames_min:
                break
            win = np.hanning(self.frame_len)
            l_hf = sosfilt(self.sos_ild, frame[:, 0] * win)
            r_hf = sosfilt(self.sos_ild, frame[:, 1] * win)
            ild = 10 * np.log10((np.sum(l_hf**2) + 1e-10) / (np.sum(r_hf**2) + 1e-10))
            ild_values.append(ild)
        return np.mean(ild_values) > 0

    def _phase2_detect_fb(self, stereo: np.ndarray, source_left: bool,
                           start_frame: int = None,
                           rot_right: bool = True) -> bool:
        """ITD slope during rotation determines front/back.

        rot_right=True: head rotates RIGHT. Positive slope → back.
        """
        if start_frame is None:
            start_frame = self.p2_start
        itd_vals = []
        n_frames = self.n_frames(len(stereo))
        end_frame = min(start_frame + self.p2_frames, n_frames)

        nfft = 1
        while nfft < 2 * self.frame_len:
            nfft *= 2

        for idx, (frame, _) in enumerate(self._make_frames(stereo)):
            if idx < start_frame:
                continue
            if idx >= end_frame:
                break
            win = np.hanning(self.frame_len)
            l_itd = sosfilt(self.sos_itd, frame[:, 0] * win)
            r_itd = sosfilt(self.sos_itd, frame[:, 1] * win)
            X = np.fft.rfft(l_itd, n=nfft) * np.conj(np.fft.rfft(r_itd, n=nfft))
            fmask = (np.fft.rfftfreq(nfft, d=1.0/self.fs) >= self.itd_band[0]) & \
                    (np.fft.rfftfreq(nfft, d=1.0/self.fs) <= self.itd_band[1])
            mag = np.abs(X)
            soft_mag = np.sqrt(mag)
            with np.errstate(divide="ignore", invalid="ignore"):
                X_soft = np.divide(X, mag, where=mag > 1e-10) * soft_mag
                X_soft[mag <= 1e-10] = 0
            X_soft[~fmask] = 0
            gcc = np.fft.irfft(X_soft, n=nfft)
            search = np.concatenate([gcc[-self.max_lag:], gcc[:self.max_lag+1]])
            peak_int = np.argmax(search)
            peak_sub = _subsample_peak(search, peak_int)
            itd_s = (peak_sub - self.max_lag) / self.fs
            itd_vals.append(itd_s)

        if len(itd_vals) < 3:
            return False

        # Linear fit of ITD vs rotation angle
        d_yaw_frame = np.deg2rad(15.0 / self.p2_frames)  # 15 deg over p2_frames
        yaw_vals = np.arange(len(itd_vals)) * d_yaw_frame
        slope = np.polyfit(yaw_vals, itd_vals, 1)[0]  # d(ITD)/d(yaw) in s/rad

        if rot_right:
            is_back = slope > 0
        else:
            is_back = slope < 0
        if self.verbose:
            print(f'  [Phase2] slope={slope*1e6:.1f}us/rad is_back={is_back}')
        return is_back
