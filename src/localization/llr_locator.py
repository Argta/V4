"""LLR-driven binaural localizer — continuous evidence accumulation for front/back.

Replaces the one-shot hard-decision Phase 2 with a log-likelihood ratio tracker
that accumulates evidence frame-by-frame from ITD predictions during head rotation.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from .base import LocalizationAlgorithm, LocalizationResult
from .xcorr_itd import itd_to_azimuth, _subsample_peak, inverse_woodworth
from .llr_core import (
    update_llr, compute_adaptive_threshold, check_stagnation,
    predict_itd, predict_itd_during_rotation,
)


class LLRLocator(LocalizationAlgorithm):
    """LLR-based localization: continuous FB evidence accumulation."""

    def __init__(self, fs: int, frame_duration_ms: float = 50.0,
                 frame_hop_ms: float = 25.0,
                 max_itd_ms: float = 1.0,
                 head_radius: float = 0.09,
                 freq_range: tuple = None,
                 head_yaw_speed: float = 60.0,
                 llr_base_threshold: float = 3.0,
                 stagnation_timeout_ms: float = 500.0,
                 enable_spectral: bool = False,
                 verbose: bool = True):
        super().__init__(fs, frame_duration_ms, frame_hop_ms)
        self.max_lag = int(max_itd_ms / 1000.0 * fs)
        self.head_radius = head_radius
        self.itd_band = freq_range or (300, 3000)
        self.ild_band = (2000, 4000)
        self.head_yaw_speed = head_yaw_speed  # deg/s, used for ITD prediction
        self.llr_base_threshold = llr_base_threshold
        self.stagnation_timeout_frames = int(stagnation_timeout_ms / frame_hop_ms)
        self.enable_spectral = enable_spectral
        self.verbose = verbose

        # Pre-design filters
        nyq = fs / 2
        self.sos_itd = butter(4, [self.itd_band[0] / nyq,
                              min(self.itd_band[1] / nyq, 0.98)],
                              btype='bandpass', output='sos')
        self.sos_ild = butter(2, [min(self.ild_band[0] / nyq, 0.95),
                                  min(self.ild_band[1] / nyq, 0.98)],
                              btype='bandpass', output='sos')

        # Phase timing (still-frames for ILD, then rotation for ITD slope)
        self.p1_frames = 4           # ILD collection (0-100ms, head still)
        self.p2_start = 4            # rotation starts

    @property
    def name(self) -> str:
        return "llr"

    def localize(self, stereo: np.ndarray) -> LocalizationResult:
        n_frames = self.n_frames(len(stereo))
        if n_frames == 0:
            return LocalizationResult(
                doa_estimated=np.array([]), timestamps=np.array([]),
                method=self.name)

        # Phase 1: ILD-based left/right
        source_left = self._phase1_detect_side(stereo)
        if self.verbose:
            print(f'  [LLR Phase1] source_left={source_left}')

        # Per-frame storage
        doa = np.zeros(n_frames)
        times = np.zeros(n_frames)
        diag_itd = np.zeros(n_frames)
        diag_phase = np.zeros(n_frames)
        llr_history = np.zeros(n_frames)
        llr_itd_hist = np.zeros(n_frames)
        llr_ild_hist = np.zeros(n_frames)
        llr_spec_hist = np.zeros(n_frames)
        dual_doa = np.zeros((n_frames, 2))
        doa_itd_only = np.zeros(n_frames)
        doa_ild_only = np.zeros(n_frames)

        # State
        llr = 0.0  # start with no prior evidence
        last_valid_itd = 0.0
        fb_determined = False
        fb_determined_frame = -1
        fb_is_back = False
        action_suggestion = None

        nfft = 1
        while nfft < 2 * self.frame_len:
            nfft *= 2

        for idx, (frame, center) in enumerate(self._make_frames(stereo)):
            t_frame = center / self.fs
            win = np.hanning(self.frame_len)

            # ── ITD extraction (GCC-PHAT per frame) ──
            l_itd = sosfilt(self.sos_itd, frame[:, 0] * win)
            r_itd = sosfilt(self.sos_itd, frame[:, 1] * win)

            L = np.fft.rfft(l_itd, n=nfft)
            R = np.fft.rfft(r_itd, n=nfft)
            X = L * np.conj(R)
            freqs = np.fft.rfftfreq(nfft, d=1.0 / self.fs)
            fmask = (freqs >= self.itd_band[0]) & (freqs <= self.itd_band[1])

            mag = np.abs(X)
            soft_mag = np.sqrt(mag)
            with np.errstate(divide="ignore", invalid="ignore"):
                X_soft = np.divide(X, mag, where=mag > 1e-10) * soft_mag
                X_soft[mag <= 1e-10] = 0
            X_soft[~fmask] = 0
            gcc = np.fft.irfft(X_soft, n=nfft)
            search = np.concatenate([gcc[-self.max_lag:], gcc[:self.max_lag + 1]])
            peak_int = np.argmax(search)
            peak_sub = _subsample_peak(search, peak_int)
            itd_s = (peak_sub - self.max_lag) / self.fs

            frame_rms = np.sqrt(np.mean(l_itd ** 2 + r_itd ** 2))
            if frame_rms < 1e-4:
                itd_s = last_valid_itd
            else:
                last_valid_itd = itd_s

            # ── ILD extraction ──
            l_hf = sosfilt(self.sos_ild, frame[:, 0] * win)
            r_hf = sosfilt(self.sos_ild, frame[:, 1] * win)
            ild_db = float(10 * np.log10(
                (np.sum(l_hf ** 2) + 1e-10) / (np.sum(r_hf ** 2) + 1e-10)))

            # ── DOA estimation ──
            fb_frame = np.column_stack([l_itd, r_itd])
            fb_fullband = np.column_stack([frame[:, 0] * win, frame[:, 1] * win])
            prev = None if idx == 0 else doa[idx - 1]

            # Raw DOA (ITD-based, unsigned lateral)
            lateral = inverse_woodworth(abs(itd_s), self.head_radius)
            lateral_deg = np.rad2deg(lateral)
            sign = -1 if itd_s < 0 else 1
            raw_az = sign * lateral_deg

            # ITD-only and ILD-only for diagnostics
            doa_itd_only[idx] = itd_to_azimuth(
                itd_s, self.head_radius,
                stereo_frame=fb_frame, fs=self.fs,
                freq_range=self.itd_band,
                fb_fullband=fb_fullband, prev_doa=prev)
            doa_ild_only[idx] = sign * lateral_deg  # no FB correction

            # Dual hypothesis DOA
            dual_doa[idx, 0] = raw_az  # H_front: use raw azimuth
            if raw_az >= 0:
                dual_doa[idx, 1] = 180.0 - raw_az  # H_back
            else:
                dual_doa[idx, 1] = -180.0 - raw_az

            # ── LLR update (during rotation) ──
            if idx >= self.p2_start and not fb_determined:
                yaw_deg = self.head_yaw_speed * (
                    t_frame - self.p2_start * self.frame_hop / self.fs)

                llr_spec = 0.0  # spectral not implemented yet
                llr, comps = update_llr(
                    llr, itd_s, ild_db, lateral_deg, yaw_deg,
                    source_left, self.head_radius,
                    llr_spectral=llr_spec)
                llr_itd_hist[idx] = comps["llr_itd"]
                llr_ild_hist[idx] = comps["llr_ild"]
                llr_spec_hist[idx] = comps["llr_spectral"]

                # Adaptive threshold
                threshold = compute_adaptive_threshold(
                    lateral_deg, yaw_deg, self.head_radius,
                    base_threshold=self.llr_base_threshold)

                if llr > threshold:
                    fb_determined = True
                    fb_determined_frame = idx
                    fb_is_back = True
                    if self.verbose:
                        print(f'  [LLR] BACK locked at frame {idx}, '
                              f'llr={llr:.2f}, thresh={threshold:.2f}')
                elif llr < -threshold:
                    fb_determined = True
                    fb_determined_frame = idx
                    fb_is_back = False
                    if self.verbose:
                        print(f'  [LLR] FRONT locked at frame {idx}, '
                              f'llr={llr:.2f}, thresh={threshold:.2f}')

            elif idx >= self.p2_start and fb_determined:
                # Tracking mode: continue accumulating but don't re-decide
                yaw_deg = self.head_yaw_speed * (
                    t_frame - self.p2_start * self.frame_hop / self.fs)
                llr_spec = 0.0
                llr, comps = update_llr(
                    llr, itd_s, ild_db, lateral_deg, yaw_deg,
                    source_left, self.head_radius,
                    llr_spectral=llr_spec)
                llr_itd_hist[idx] = comps["llr_itd"]
                llr_ild_hist[idx] = comps["llr_ild"]
                llr_spec_hist[idx] = comps["llr_spectral"]

            # ── Stagnation check ──
            if not fb_determined and idx >= self.p2_start:
                if check_stagnation(
                    llr_history[:idx + 1], self.stagnation_timeout_frames):
                    action_suggestion = "perturb_rotation"
                    if self.verbose:
                        print(f'  [LLR] STAGNATION at frame {idx}, '
                              f'suggesting perturbation')

            # ── Final DOA with FB correction ──
            doa[idx] = itd_to_azimuth(
                itd_s, self.head_radius,
                stereo_frame=fb_frame, fs=self.fs,
                freq_range=self.itd_band,
                fb_fullband=fb_fullband,
                prev_doa=prev,
                fb_active=fb_determined,
                fb_is_back=fb_is_back)

            times[idx] = t_frame
            diag_itd[idx] = itd_s
            diag_phase[idx] = ild_db
            llr_history[idx] = llr

        # Median filter cleanup
        from scipy.signal import medfilt
        doa = medfilt(doa, kernel_size=5)
        doa_itd_only = medfilt(doa_itd_only, kernel_size=5)
        doa_ild_only = medfilt(doa_ild_only, kernel_size=5)

        return LocalizationResult(
            doa_estimated=doa,
            timestamps=times,
            method=self.name,
            itd_per_frame=diag_itd,
            phase_mean=diag_phase,
            doa_itd_only=doa_itd_only,
            doa_ild_only=doa_ild_only,
            llr_history=llr_history,
            llr_components={
                "itd": llr_itd_hist,
                "ild": llr_ild_hist,
                "spectral": llr_spec_hist,
            },
            fb_determined=fb_determined,
            fb_determined_frame=fb_determined_frame,
            fb_is_back=fb_is_back,
            action_suggestion=action_suggestion,
            dual_hypothesis_doa=dual_doa,
        )

    # ---- Streaming API (Phase 1) ----

    def reset(self):
        """Clear internal state for a new streaming session."""
        super().reset()
        self._llr_value = 0.0
        self._fb_determined = False
        self._fb_is_back = False
        self._source_left = None       # None = not yet determined
        self._p1_ild_values = []
        self._last_valid_itd = 0.0
        self._yaw_accumulated = 0.0    # accumulated rotation for ITD prediction
        self._nfft = 1
        while self._nfft < 2 * self.frame_len:
            self._nfft *= 2
        self._accum_time = 0.0

    def process_frame(self, frame, yaw_head=0.0):
        """Process a single stereo frame with LLR accumulation.

        Args:
            frame: (frame_len, 2) stereo samples
            yaw_head: current head yaw (deg), 0=faces forward

        Returns:
            (doa_world_deg, confidence) tuple
        """
        from .llr_core import update_llr, compute_adaptive_threshold, check_stagnation

        # -- NFFT (lazy) --
        if not hasattr(self, "_nfft"):
            self._nfft = 1
            while self._nfft < 2 * self.frame_len:
                self._nfft *= 2
        nfft = self._nfft

        win = np.hanning(self.frame_len)
        idx = self._frame_idx

        # -- ITD extraction (GCC-PHAT per frame) --
        l_itd = sosfilt(self.sos_itd, frame[:, 0] * win)
        r_itd = sosfilt(self.sos_itd, frame[:, 1] * win)

        from .gcc_phat import _gcc_phat_single_frame
        itd_s, gcc, freqs, L, R = _gcc_phat_single_frame(
            frame, win, self.max_lag, nfft,
            self.sos_itd, self.itd_band, self.fs)

        # Energy gate
        frame_rms = np.sqrt(np.mean(l_itd ** 2 + r_itd ** 2))
        if frame_rms < 1e-4:
            itd_s = getattr(self, "_last_valid_itd", 0.0)
        else:
            self._last_valid_itd = itd_s

        # -- ILD --
        l_hf = sosfilt(self.sos_ild, frame[:, 0] * win)
        r_hf = sosfilt(self.sos_ild, frame[:, 1] * win)
        ild_db = 10 * np.log10((np.sum(l_hf ** 2) + 1e-10) / (np.sum(r_hf ** 2) + 1e-10))

        # Lateral angle from ITD
        from .xcorr_itd import inverse_woodworth, itd_to_azimuth
        lateral_deg = np.rad2deg(inverse_woodworth(abs(itd_s), self.head_radius))

        # -- Phase 1: ILD collection for left/right --
        if self._source_left is None and idx < self.p1_frames:
            self._p1_ild_values.append(ild_db)
        if idx == self.p1_frames - 1 and self._source_left is None:
            self._source_left = float(np.mean(self._p1_ild_values)) > 0

        # -- Front/back frame data --
        fb_frame = np.column_stack([l_itd, r_itd])
        fb_fullband = np.column_stack([frame[:, 0] * win, frame[:, 1] * win])

        # -- LLR update (during rotation phase) --
        yaw_deg = yaw_head  # current head yaw
        if idx >= self.p2_start and self._source_left is not None and not self._fb_determined:
            llr_spec = 0.0
            self._llr_value, comps = update_llr(
                self._llr_value, itd_s, ild_db, lateral_deg, yaw_deg,
                self._source_left, self.head_radius,
                llr_spectral=llr_spec)

            # Adaptive threshold
            threshold = compute_adaptive_threshold(
                lateral_deg, yaw_deg, self.head_radius,
                base_threshold=self.llr_base_threshold)

            if self._llr_value > threshold:
                self._fb_determined = True
                self._fb_is_back = True
            elif self._llr_value < -threshold:
                self._fb_determined = True
                self._fb_is_back = False

        elif idx >= self.p2_start and self._source_left is not None and self._fb_determined:
            # Tracking mode: continue accumulating
            llr_spec = 0.0
            self._llr_value, _ = update_llr(
                self._llr_value, itd_s, ild_db, lateral_deg, yaw_deg,
                self._source_left, self.head_radius,
                llr_spectral=llr_spec)

        # -- DOA computation --
        prev = self._prev_doa
        doa_head = itd_to_azimuth(
            itd_s, self.head_radius,
            stereo_frame=fb_frame, fs=self.fs,
            freq_range=self.itd_band,
            fb_fullband=fb_fullband,
            prev_doa=prev,
            fb_active=self._fb_determined,
            fb_is_back=self._fb_is_back)

        # World-frame
        doa_world = doa_head + yaw_head
        doa_world = ((doa_world + 180) % 360) - 180

        # Confidence
        search = np.concatenate([gcc[-self.max_lag:], gcc[:self.max_lag + 1]])
        peak_val = float(np.max(search))
        conf = min(1.0, max(0.0, peak_val / (frame_rms * 5 + 1e-10)))

        # -- Update state --
        self._prev_doa = doa_head
        self._frame_idx += 1
        self._accum_time += self.frame_hop / self.fs

        return doa_world, conf


    def _phase1_detect_side(self, stereo: np.ndarray) -> bool:
        """ILD-based left/right detection. True = source LEFT."""
        n_frames = min(self.n_frames(len(stereo)), self.p1_frames)
        if n_frames < 2:
            return False
        ild_values = []
        for idx, (frame, _) in enumerate(self._make_frames(stereo)):
            if idx >= n_frames:
                break
            win = np.hanning(self.frame_len)
            l_hf = sosfilt(self.sos_ild, frame[:, 0] * win)
            r_hf = sosfilt(self.sos_ild, frame[:, 1] * win)
            ild = 10 * np.log10(
                (np.sum(l_hf ** 2) + 1e-10) / (np.sum(r_hf ** 2) + 1e-10))
            ild_values.append(ild)
        return float(np.mean(ild_values)) > 0