"""Bowl impact sound - transient impact with structural modal response."""

import numpy as np


def generate(t, fs, cfg):
    """Generate bowl impact sound.

    cfg params:
        impact_duration: float (default 0.005) - impact pulse width in seconds
        modal_freqs: list[float] - structural modal frequencies in Hz
        reflection_delay: float (default 0.002) - ground reflection delay in seconds
    """
    impact_duration = cfg.get("impact_duration", 0.005)
    modal_freqs = cfg.get("modal_freqs", [800, 1600, 2400, 3200, 4200, 5600, 7000])
    reflection_delay = cfg.get("reflection_delay", 0.002)
    impact_len = int(impact_duration * fs)

    # Impact excitation - half-sine pulse
    impact = np.sin(np.linspace(0, np.pi, impact_len))

    # Structural modal response
    bowl_vib = np.zeros(len(t))
    for i, freq in enumerate(modal_freqs):
        omega = 2 * np.pi * freq
        decay = np.exp(-15 * t)
        response = decay * np.sin(omega * t)
        temp = np.convolve(impact, response, mode="same")
        L = min(len(temp), len(bowl_vib))
        amplitude = 1 / ((i + 1) ** 0.7)
        bowl_vib[:L] += temp[:L] * amplitude

    # Acoustic radiation: direct + ground reflection
    direct = bowl_vib.copy()
    delay = int(reflection_delay * fs)
    reflection = np.zeros(len(t))
    if delay < len(t):
        reflection[delay:] = direct[:-delay] * 0.5
    bowl_sound = direct + reflection

    # High-frequency click transient
    click = np.exp(-200 * t[:impact_len]) * np.sin(2 * np.pi * 8000 * t[:impact_len])
    bowl_sound[:impact_len] += click * 0.3

    # Normalize
    mx = np.max(np.abs(bowl_sound))
    if mx > 0:
        bowl_sound /= mx
        bowl_sound *= 0.9

    return bowl_sound
