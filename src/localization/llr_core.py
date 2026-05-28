"""LLR core — pure mathematical functions for front/back log-likelihood ratio.

No I/O, no classes, no external state. All functions are deterministic given inputs.
"""

import numpy as np

SPEED_OF_SOUND = 343.0


# ── Woodworth ITD prediction ──────────────────────────────────────────────

def predict_itd(azimuth_deg: float, head_yaw_deg: float,
                head_radius: float = 0.09,
                sound_speed: float = SPEED_OF_SOUND) -> float:
    """Predict ITD for a given source azimuth and head yaw using Woodworth model.

    Args:
        azimuth_deg: source azimuth in head-local coordinates (0=ahead, +=right)
        head_yaw_deg: head yaw angle (unused in head-local, kept for interface)
        head_radius: head radius in meters
        sound_speed: speed of sound in m/s

    Returns:
        predicted ITD in seconds (positive = right ear leads)
    """
    theta = np.deg2rad(abs(azimuth_deg))
    if theta > np.pi / 2:
        theta = np.pi - theta  # fold back hemisphere

    itd_mag = (head_radius / sound_speed) * (theta + np.sin(theta))
    sign = 1 if azimuth_deg >= 0 else -1
    return sign * itd_mag


# ── Dual-hypothesis ITD prediction ────────────────────────────────────────

def predict_itd_dual(lateral_deg: float, head_radius: float = 0.09,
                     sound_speed: float = SPEED_OF_SOUND):
    """Compute predicted ITD for front and back hypotheses at same lateral angle.

    lateral_deg: magnitude of lateral angle [0, 90]
        Front: az = +lateral  -> ITD = +itd_mag
        Back:  az = 180 - lateral (right) -> after fold: same magnitude, same sign

    Actually both front and back give the SAME Woodworth ITD magnitude.
    The difference emerges only during head rotation.

    Returns:
        (itd_front, itd_back) in seconds
    """
    lat_rad = np.deg2rad(lateral_deg)
    itd_mag = (head_radius / sound_speed) * (lat_rad + np.sin(lat_rad))
    # Both front and back have same |ITD|; sign from left/right (already determined)
    return itd_mag, itd_mag  # magnitude only — sign comes from ILD


def predict_itd_during_rotation(lateral_deg: float, head_yaw_deg: float,
                                source_left: bool,
                                head_radius: float = 0.09,
                                sound_speed: float = SPEED_OF_SOUND):
    """Predict ITD for front and back hypotheses during head rotation.

    When head rotates RIGHT:
    - Front source: ITD magnitude DECREASES (source approaches midline)
    - Back source:  ITD magnitude INCREASES (source moves away from midline)

    Args:
        lateral_deg: lateral angle magnitude [0, 90] at current head yaw
        head_yaw_deg: current head yaw (positive = right)
        source_left: True if source is on left side
        head_radius: head radius
        sound_speed: speed of sound

    Returns:
        (itd_front, itd_back) predicted ITD in seconds for each hypothesis
    """
    # Effective azimuth under each hypothesis in world frame, then head-local
    lat = lateral_deg

    # Front hypothesis: azimuth = +lat (right) or -lat (left)
    az_front = -lat if source_left else lat
    az_front_local = az_front - head_yaw_deg

    # Back hypothesis: azimuth = 180 - lat (right back) or -180 + lat (left back)
    if source_left:
        az_back = -180 + lat
    else:
        az_back = 180 - lat
    az_back_local = az_back - head_yaw_deg

    itd_front = predict_itd(az_front_local, 0, head_radius, sound_speed)
    itd_back = predict_itd(az_back_local, 0, head_radius, sound_speed)

    return itd_front, itd_back


# ── Discrimination power ──────────────────────────────────────────────────

def compute_discrimination_power(lateral_deg: float, head_yaw_deg: float,
                                 head_radius: float = 0.09,
                                 sigma_itd: float = 20e-6) -> float:
    """Compute how well front and back can be distinguished at current state.

    Returns the ITD difference between hypotheses in units of sigma_itd.
    Higher value = easier to discriminate.

    Args:
        lateral_deg: lateral angle [0, 90]
        head_yaw_deg: head yaw in degrees
        head_radius: head radius
        sigma_itd: ITD measurement noise std in seconds (~20us for GCC-PHAT)

    Returns:
        discrimination power (dimensionless)
    """
    if abs(lateral_deg) < 2.0:
        return 0.0  # Near median plane, can't discriminate

    itd_f, itd_b = predict_itd_during_rotation(
        abs(lateral_deg), head_yaw_deg, lateral_deg < 0, head_radius)
    return abs(itd_f - itd_b) / max(sigma_itd, 1e-12)


# ── Adaptive threshold ────────────────────────────────────────────────────

def compute_adaptive_threshold(lateral_deg: float, head_yaw_deg: float,
                               head_radius: float = 0.09,
                               sigma_itd: float = 20e-6,
                               base_threshold: float = 3.0) -> float:
    """Compute adaptive LLR threshold based on current discriminability.

    When source is far lateral (large ITD), threshold is low → fast lock.
    When source is near median plane (small ITD), threshold is high → cautious.

    Args:
        lateral_deg: lateral angle [0, 90]
        head_yaw_deg: head yaw in degrees
        head_radius: head radius
        sigma_itd: ITD noise std
        base_threshold: base LLR threshold

    Returns:
        adaptive threshold
    """
    disc_power = compute_discrimination_power(
        lateral_deg, head_yaw_deg, head_radius, sigma_itd)
    return base_threshold / max(disc_power, 0.2)


# ── LLR update ────────────────────────────────────────────────────────────

def update_llr(llr_prev: float,
               itd_observed: float,
               ild_observed: float,
               lateral_deg: float,
               head_yaw_deg: float,
               source_left: bool,
               head_radius: float = 0.09,
               sigma_itd: float = 30e-6,
               sigma_ild: float = 5.0,
               llr_spectral: float = 0.0) -> tuple:
    """Compute single-frame LLR increment for front/back hypotheses.

    Returns (new_llr, components_dict) where components = {
        "llr_itd": float, "llr_ild": float, "llr_spectral": float, "total": float
    }

    Args:
        llr_prev: previous LLR value
        itd_observed: measured ITD in seconds (signed)
        ild_observed: measured ILD in dB (positive = left louder)
        lateral_deg: lateral angle magnitude [0, 90]
        head_yaw_deg: current head yaw
        source_left: whether source is on left side
        head_radius: head radius
        sigma_itd: ITD measurement noise std
        sigma_ild: ILD measurement noise std (dB)
        llr_spectral: external spectral LLR contribution

    Returns:
        (new_llr, components)
    """
    if lateral_deg < 2.0:
        # Too close to median plane — no meaningful FB discrimination
        components = {"llr_itd": 0.0, "llr_ild": 0.0,
                      "llr_spectral": llr_spectral, "total": llr_spectral}
        return llr_prev + llr_spectral, components

    # Predict ITD under both hypotheses
    itd_f, itd_b = predict_itd_during_rotation(
        lateral_deg, head_yaw_deg, source_left, head_radius)

    # Log-likelihood: P(obs | H_back) - P(obs | H_front)
    # Using Gaussian observation model: log P = -0.5 * ((obs - pred) / sigma)^2
    llr_itd = -0.5 * ((itd_observed - itd_b) / sigma_itd) ** 2
    llr_itd += 0.5 * ((itd_observed - itd_f) / sigma_itd) ** 2

    # ILD: back sources typically have slightly different ILD patterns
    # Simplistic model: no strong ILD difference between front/back for broadband
    # But we keep the term for future refinement
    llr_ild = 0.0

    total_delta = llr_itd + llr_ild + llr_spectral
    components = {
        "llr_itd": llr_itd,
        "llr_ild": llr_ild,
        "llr_spectral": llr_spectral,
        "total": total_delta,
    }
    return llr_prev + total_delta, components


# ── Stagnation detection ──────────────────────────────────────────────────

def check_stagnation(llr_history: np.ndarray, window_frames: int = 20,
                     stagnation_threshold: float = 0.5) -> bool:
    """Check if LLR has stagnated (not converging to any decision).

    Stagnation criteria:
    1. |LLR| < stagnation_threshold for all frames in recent window
    2. Variance of LLR in window is low

    Args:
        llr_history: array of recent LLR values
        window_frames: number of frames to check
        stagnation_threshold: max |LLR| to consider "not decided"

    Returns:
        True if stagnated
    """
    if len(llr_history) < window_frames:
        return False

    recent = llr_history[-window_frames:]
    all_near_zero = np.all(np.abs(recent) < stagnation_threshold)
    low_variance = np.var(recent) < 0.1

    return all_near_zero and low_variance