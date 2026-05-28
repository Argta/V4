"""Noise models for binaural localization evaluation."""

import numpy as np


def white_noise(shape, rng=None):
    """Generate white Gaussian noise."""
    if rng is None:
        rng = np.random
    return rng.randn(*shape).astype(np.float64)


def pink_noise(n_samples, rng=None):
    """Generate approximate pink (1/f) noise via Voss-McCartney."""
    if rng is None:
        rng = np.random
    n_octaves = 16
    pink = np.zeros(n_samples, dtype=np.float64)
    for octave in range(n_octaves):
        step = 1 << octave
        n = (n_samples + step - 1) // step
        noise = rng.randn(n).astype(np.float64)
        for i, val in enumerate(noise):
            pink[i * step] += val
    pink /= np.sqrt(n_octaves)
    return pink


def background_noise(stereo, snr_db, noise_type="white", rng=None):
    """Add background (uncorrelated) noise to stereo signal.

    Each channel gets independent noise at the specified SNR.

    Args:
        stereo: (N, 2) stereo signal
        snr_db: signal-to-noise ratio in dB
        noise_type: "white" or "pink"
        rng: optional numpy random generator

    Returns:
        (N, 2) noisy signal
    """
    if rng is None:
        rng = np.random

    signal_power = np.mean(stereo ** 2)
    if signal_power < 1e-12:
        return stereo

    desired_noise_power = signal_power / (10 ** (snr_db / 10))

    n = len(stereo)
    if noise_type == "pink":
        noise_l = pink_noise(n, rng)
        noise_r = pink_noise(n, rng)
    else:
        noise_l = rng.randn(n).astype(np.float64)
        noise_r = rng.randn(n).astype(np.float64)

    # Scale to desired power
    noise_l *= np.sqrt(desired_noise_power / (np.mean(noise_l ** 2) + 1e-12))
    noise_r *= np.sqrt(desired_noise_power / (np.mean(noise_r ** 2) + 1e-12))

    return stereo + np.column_stack([noise_l, noise_r])


def sensor_noise(stereo, snr_db, rng=None):
    """Add sensor (microphone) noise — identical noise model for both channels.

    Args:
        stereo: (N, 2) stereo signal
        snr_db: SNR in dB
        rng: numpy random generator

    Returns:
        (N, 2) noisy signal
    """
    if rng is None:
        rng = np.random

    signal_power = np.mean(stereo ** 2)
    if signal_power < 1e-12:
        return stereo

    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.randn(len(stereo)).astype(np.float64)
    noise *= np.sqrt(desired_noise_power / (np.mean(noise ** 2) + 1e-12))

    return stereo + np.column_stack([noise, noise])


def directional_noise(stereo, azimuth_deg, snr_db, head_radius=0.09,
                      fs=44100, rng=None):
    """Add directional interfering source at specified azimuth.

    A secondary source with independent noise placed at the given direction.

    Args:
        stereo: (N, 2) stereo signal
        azimuth_deg: direction of interferer
        snr_db: signal-to-interference ratio
        head_radius: head radius for ITD modeling
        fs: sample rate
        rng: numpy random generator

    Returns:
        (N, 2) signal with directional interferer
    """
    if rng is None:
        rng = np.random

    signal_power = np.mean(stereo ** 2)
    if signal_power < 1e-12:
        return stereo

    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.randn(len(stereo)).astype(np.float64)
    noise *= np.sqrt(desired_noise_power / (np.mean(noise ** 2) + 1e-12))

    # Apply ITD for the interferer direction
    from src.spatial.hrtf import compute_itd
    theta = abs(np.deg2rad(azimuth_deg))
    theta = min(theta, np.pi / 2)
    itd_s = compute_itd(theta, head_radius)
    itd_samples = int(round(itd_s * fs))

    if azimuth_deg >= 0:
        # Source on right: right ear leads
        noise_left = np.zeros_like(noise)
        noise_left[itd_samples:] = noise[:len(noise) - itd_samples] if itd_samples > 0 else noise
        noise_right = noise.copy()
    else:
        # Source on left: left ear leads
        noise_left = noise.copy()
        noise_right = np.zeros_like(noise)
        noise_right[itd_samples:] = noise[:len(noise) - itd_samples] if itd_samples > 0 else noise

    return stereo + np.column_stack([noise_left, noise_right])
