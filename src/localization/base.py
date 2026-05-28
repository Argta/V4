"""Base classes and result container for localization algorithms."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class LocalizationResult:
    """Container for localization algorithm output."""
    doa_estimated: np.ndarray        # (M,) estimated azimuth in degrees
    timestamps: np.ndarray           # (M,) center time of each frame in seconds
    frame_indices: Optional[np.ndarray] = None  # (M,) sample index of each frame
    confidence: Optional[np.ndarray] = None     # (M,) per-frame confidence score
    method: str = ""
    # Diagnostic fields for debugging
    itd_per_frame: Optional[np.ndarray] = None      # (M,) ITD in seconds
    lateral_angle: Optional[np.ndarray] = None      # (M,) lateral angle [0,90] deg
    phase_mean: Optional[np.ndarray] = None         # (M,) xcorr peak height (confidence)
    freq_mean: Optional[np.ndarray] = None          # (M,) dominant frequency Hz
    doa_itd_only: Optional[np.ndarray] = None     # (M,) ITD-only DOA (before ILD fusion)
    doa_ild_only: Optional[np.ndarray] = None     # (M,) ILD-only DOA (from theta_ild)
    # Phase 4 lock-on-source
    lock_achieved: bool = False                     # whether lock was achieved
    lock_frame: int = -1                            # frame index where lock first detected
    doa_smoothed: Optional[np.ndarray] = None        # (M,) smoothed DOA after lock
    # Single-frame detail for step-by-step analysis
    diag_frame_left: Optional[np.ndarray] = None    # (L,) bandpass-filtered left
    diag_frame_right: Optional[np.ndarray] = None   # (L,) bandpass-filtered right
    diag_freqs: Optional[np.ndarray] = None         # (K,) frequency bins
    diag_spec_left: Optional[np.ndarray] = None     # (K,) |L| magnitude
    diag_spec_right: Optional[np.ndarray] = None    # (K,) |R| magnitude
    diag_xphase: Optional[np.ndarray] = None        # (Q,) cross-correlation function
    diag_xweight: Optional[np.ndarray] = None       # (Q,) xcorr lag axis (us)
    # v4.0 LLR fields
    llr_history: Optional[np.ndarray] = None        # (M,) LLR value per frame
    llr_components: Optional[dict] = None           # {"itd": array, "ild": array, "spectral": array}
    fb_determined: bool = False                     # whether front/back has been decided
    fb_determined_frame: int = -1                   # frame index where FB decision made
    fb_is_back: bool = False                        # result of FB decision
    action_suggestion: Optional[str] = None          # None | "perturb_rotation" | "panic_turn"
    dual_hypothesis_doa: Optional[np.ndarray] = None # (M,2) [front_doa, back_doa] per frame


class LocalizationAlgorithm:
    """Abstract base class for binaural localization algorithms."""

    def __init__(self, fs: int, frame_duration_ms: float = 50.0,
                 frame_hop_ms: float = 25.0):
        self.fs = fs
        self.frame_len = int(frame_duration_ms / 1000.0 * fs)
        self.frame_hop = int(frame_hop_ms / 1000.0 * fs)

    @property
    def name(self) -> str:
        raise NotImplementedError

    def localize(self, stereo: np.ndarray) -> LocalizationResult:
        """Estimate DOA from stereo binaural signal.

        Args:
            stereo: (N, 2) stereo signal

        Returns:
            LocalizationResult with estimated DOA per frame
        """
        raise NotImplementedError

    def _make_frames(self, stereo: np.ndarray):
        """Yield (frame, center_sample_index) tuples from stereo signal."""
        n = len(stereo)
        for start in range(0, n - self.frame_len + 1, self.frame_hop):
            end = start + self.frame_len
            center = start + self.frame_len // 2
            yield stereo[start:end], center

    def n_frames(self, signal_len: int) -> int:
        """Number of frames for a signal of given length."""
        if signal_len < self.frame_len:
            return 0
        return (signal_len - self.frame_len) // self.frame_hop + 1