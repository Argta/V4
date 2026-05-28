"""Chair sliding sound - modulated friction noise with structural modes."""

import numpy as np
from scipy.signal import butter, lfilter


def _butter_bandpass(low, high, fs, order=2):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def generate(t, fs, cfg):
    """Generate chair sliding sound.

    cfg params:
        friction_freq: float (default 50) - friction modulation frequency in Hz
        modal_freqs: list[float] - structural modal frequencies in Hz
    """
    friction_freq = cfg.get("friction_freq", 50)
    modal_freqs = cfg.get("modal_freqs", [80, 150, 320, 600, 1200, 2500])

    # Friction excitation: modulated noise
    friction_signal = np.zeros(len(t))
    for i in range(len(t)):
        envelope = (np.sin(2 * np.pi * friction_freq * t[i]) + 1) / 2
        envelope = envelope ** 2
        friction_signal[i] = envelope * np.random.randn()

    # Smoothing
    window = 20
    friction_signal = np.convolve(
        friction_signal, np.ones(window) / window, mode="same"
    )

    # Structural modal response via bandpass filtering
    chair_resp = friction_signal.copy()
    for f0 in modal_freqs:
        bw = f0 * 0.15
        low = max(10, f0 - bw / 2)
        high = min(fs / 2 - 10, f0 + bw / 2)
        b, a = _butter_bandpass(low, high, fs)
        modal = lfilter(b, a, friction_signal)
        chair_resp += modal * 0.3

    # Normalize
    mx = np.max(np.abs(chair_resp))
    if mx > 0:
        chair_resp /= mx
        chair_resp *= 0.7

    return chair_resp
