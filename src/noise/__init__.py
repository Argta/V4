"""Noise injection for binaural signals."""

import numpy as np

from .models import background_noise, sensor_noise, directional_noise


def add_noise(stereo, fs, noise_config):
    """Add noise to stereo signal according to configuration.

    Args:
        stereo: (N, 2) stereo signal
        fs: sample rate
        noise_config: NoiseConfig from schema

    Returns:
        (N, 2) noisy stereo signal
    """
    if not noise_config.enabled:
        return stereo

    rng = np.random.RandomState(42)
    noisy = stereo.copy()

    # Background noise (uncorrelated per channel)
    if noise_config.background_snr_db < 100:
        noisy = background_noise(
            noisy, noise_config.background_snr_db,
            noise_type=noise_config.noise_type, rng=rng
        )

    # Sensor noise (correlated across channels)
    if noise_config.sensor_snr_db < 100:
        noisy = sensor_noise(noisy, noise_config.sensor_snr_db, rng=rng)

    return noisy
