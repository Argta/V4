"""Localization algorithms for binaural DOA estimation."""

from .ground_truth import compute_truth_doa, compute_truth_elevation
from .base import LocalizationResult, LocalizationAlgorithm, FrameResult
from .xcorr_itd import XCorrITD
from .gcc_phat import GCCPHAT
from .srp_phat import SRPPHAT
from .active_locator import ActiveLocator
from .llr_locator import LLRLocator


def create_localizer(method: str, fs: int, frame_duration_ms: float = 50.0,
                     frame_hop_ms: float = 25.0,
                     head_radius: float = 0.09,
                     freq_range: tuple = None,
                     head_yaw_speed: float = 0.0,
                     active_head: bool = False,
                     llr_base_threshold: float = 3.0,
                     stagnation_timeout_ms: float = 500.0,
                     enable_spectral: bool = False,
                     verbose: bool = True) -> LocalizationAlgorithm:
    """Factory function to create a localization algorithm.

    Args:
        method: "xcorr_itd" | "gcc_phat" | "srp_phat" | "llr"
        fs: sample rate
        frame_duration_ms: analysis frame length in ms
        frame_hop_ms: frame hop in ms
        head_radius: head radius in meters
        freq_range: (low, high) bandpass range
        head_yaw_speed: deg/s head rotation speed (for active_head / LLR)
        active_head: enable active head-rotation with ActiveLocator
        llr_base_threshold: LLR base threshold (for llr method)
        stagnation_timeout_ms: stagnation timeout (for llr method)
        enable_spectral: enable spectral notch detection (for llr method)
        verbose: if False, suppress diagnostic prints

    Returns:
        LocalizationAlgorithm instance
    """
    # v3.0: active_head uses ActiveLocator (flash-turn-track)
    if active_head:
        return ActiveLocator(
            fs=fs, frame_duration_ms=frame_duration_ms,
            frame_hop_ms=frame_hop_ms, head_radius=head_radius,
            freq_range=freq_range, verbose=verbose)

    methods = {
        "xcorr_itd": XCorrITD,
        "gcc_phat": GCCPHAT,
        "srp_phat": SRPPHAT,
        "llr": LLRLocator,
    }

    cls = methods.get(method)
    if cls is None:
        raise ValueError(f"Unknown localization method: {method}. "
                         f"Choose from {list(methods.keys())}")

    if method == "llr":
        return LLRLocator(
            fs=fs, frame_duration_ms=frame_duration_ms,
            frame_hop_ms=frame_hop_ms, head_radius=head_radius,
            freq_range=freq_range, head_yaw_speed=head_yaw_speed,
            llr_base_threshold=llr_base_threshold,
            stagnation_timeout_ms=stagnation_timeout_ms,
            enable_spectral=enable_spectral, verbose=verbose)

    kwargs = dict(fs=fs, frame_duration_ms=frame_duration_ms,
                  frame_hop_ms=frame_hop_ms, head_radius=head_radius,
                  head_yaw_speed=head_yaw_speed)
    if freq_range is not None:
        kwargs["freq_range"] = freq_range
    if method == "srp_phat":
        kwargs.pop("freq_range", None)
        return SRPPHAT(fs=fs, frame_duration_ms=frame_duration_ms,
                       frame_hop_ms=frame_hop_ms, head_radius=head_radius,
                       freq_range=freq_range)
    return cls(**kwargs)


def localize(stereo, fs, method="gcc_phat", **kwargs):
    """Convenience function: localize stereo signal with given method.

    Args:
        stereo: (N, 2) stereo array
        fs: sample rate
        method: algorithm name
        **kwargs: passed to create_localizer

    Returns:
        LocalizationResult
    """
    loc = create_localizer(method, fs, **kwargs)
    return loc.localize(stereo)


__all__ = [
    "compute_truth_doa", "compute_truth_elevation",
    "LocalizationResult", "LocalizationAlgorithm", "FrameResult",
    "XCorrITD", "GCCPHAT", "SRPPHAT", "ActiveLocator", "LLRLocator",
    "create_localizer", "localize",
]
