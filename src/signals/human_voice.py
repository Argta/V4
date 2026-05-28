"""Human voice - formant synthesis using glottal pulse + vocal tract filtering."""

import numpy as np
from scipy.signal import butter, lfilter


def _butter_bandpass(low, high, fs, order=2):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def generate(t, fs, cfg):
    """Generate human voice signal.

    cfg params:
        F0: float (default 120) - fundamental frequency in Hz
        formants: list[float] - formant frequencies in Hz
    """
    F0 = cfg.get("F0", 120)
    formants = cfg.get("formants", [850, 1700, 2600, 3600])
    period_samples = fs / F0

    # Glottal pulse train
    glottal = np.zeros(len(t))
    for idx in range(len(t)):
        tau = (idx % period_samples) / period_samples
        if tau < 0.1:
            glottal[idx] = np.sin(np.pi * tau / 0.1)
        elif tau < 0.5:
            glottal[idx] = np.sin(np.pi * (tau - 0.1) / 0.4)
        else:
            glottal[idx] = 0

    # Vocal tract: cascade bandpass filters at formant frequencies
    voice = glottal.copy()
    for f0 in formants:
        bw = f0 * 0.15
        low = max(10, f0 - bw / 2)
        high = min(fs / 2 - 10, f0 + bw / 2)
        b, a = _butter_bandpass(low, high, fs)
        voice = lfilter(b, a, voice)

    # Normalize
    mx = np.max(np.abs(voice))
    if mx > 0:
        voice /= mx
        voice *= 0.8

    return voice
