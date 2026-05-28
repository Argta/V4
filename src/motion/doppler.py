"""Doppler effect simulation via time-varying resampling.

For a moving source, the received frequency is shifted by:
    f' = f * c / (c - v_radial)

where v_radial is positive when the source moves toward the receiver.
This module resamples the signal to account for the time-varying
compression/stretching of the waveform.
"""

import numpy as np
from scipy.interpolate import interp1d


SPEED_OF_SOUND = 343.0  # m/s at 20°C


def apply_doppler(signal: np.ndarray, trajectory: np.ndarray,
                  receiver_pos, fs: int,
                  speed_of_sound: float = SPEED_OF_SOUND) -> np.ndarray:
    """Apply Doppler shift to a signal from a moving source.

    Args:
        signal: 1-D source signal samples
        trajectory: (N, 3) array of source positions over time
        receiver_pos: [x, y, z] receiver position (one ear)
        fs: sample rate in Hz
        speed_of_sound: speed of sound in m/s

    Returns:
        1-D Doppler-shifted signal, same length as input
    """
    if len(trajectory) < 2:
        return signal

    receiver = np.array(receiver_pos)
    n_samples = len(signal)

    # Limit trajectory to signal length
    if len(trajectory) > n_samples:
        trajectory = trajectory[:n_samples]
    else:
        n_samples = len(trajectory)
        signal = signal[:n_samples]

    # Compute distance and radial velocity for each sample
    distances = np.linalg.norm(trajectory - receiver, axis=1)

    # Smooth distances to reduce noise in velocity
    win = max(int(fs * 0.005), 3)  # 5ms window
    if win % 2 == 0:
        win += 1
    if win < len(distances):
        from scipy.ndimage import uniform_filter1d
        distances = uniform_filter1d(distances, size=win)

    # Radial velocity: negative = approaching, positive = receding
    velocity = np.gradient(distances, 1.0 / fs)
    # v_radial positive when source moves toward receiver
    v_radial = -velocity

    # Avoid division by zero when v_radial approaches c
    v_radial = np.clip(v_radial, -speed_of_sound * 0.9, speed_of_sound * 0.9)

    # Doppler factor: f' = f * c / (c - v_radial)
    doppler_factor = speed_of_sound / (speed_of_sound - v_radial)

    # Cumulative time warp: tau(t) = integral(0, t, doppler_factor(tau) dtau)
    dt = 1.0 / fs
    tau = np.cumsum(doppler_factor) * dt

    # Normalize tau to match original duration
    tau = tau * (len(signal) - 1) * dt / tau[-1] if tau[-1] > 0 else tau

    # Create output time grid (original timing)
    t_out = np.arange(n_samples) * dt

    # Interpolate signal at warped time points
    interpolator = interp1d(
        tau, signal,
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    warped = interpolator(t_out)

    return warped.astype(np.float64)
