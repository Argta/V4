"""Measured HRTF spatialization (HRTF Mode C).

Placeholder: loads SOFA-format HRIR datasets and applies them via
FIR convolution. Implementation is deferred until a suitable HRTF
dataset is available.

When a SOFA file is provided via MicrophoneConfig.hrtf_dataset:
1. Load HRIR pairs indexed by azimuth/elevation
2. For each frame, look up the nearest-direction HRIR
3. Convolve left/right signals with the corresponding HRIR

The unified interface matches hrtf.py and hrtf_parametric.py so
Mode C is a drop-in replacement once implemented.
"""

import numpy as np


def _load_sofa(filepath):
    """Load HRTF dataset from SOFA file.

    Args:
        filepath: path to .sofa file

    Returns:
        (hrir_data, source_positions, metadata) tuples

    Raises:
        NotImplementedError: until SOFA dataset is provided
    """
    raise NotImplementedError(
        "Measured HRTF mode requires a SOFA-format HRTF dataset.\n"
        "Place the .sofa file in the project and set:\n"
        "  microphone:\n"
        "    hrtf_mode: measured\n"
        "    hrtf_dataset: path/to/your_hrir.sofa\n"
        "    hrtf_subject: 0  # subject index\n"
        "\n"
        "Recommended public datasets:\n"
        "  - CIPIC:  https://www.ece.ucdavis.edu/cipic/spatial-sound/\n"
        "  - LISTEN: http://recherche.ircam.fr/equipes/salles/listen/\n"
        "  - ARI:    https://www.oeaw.ac.at/isf/das-institut/software/hrtf-database\n"
    )


def process_binaural(signal_left, signal_right,
                     source_pos, left_ear_pos, right_ear_pos,
                     head_center, head_radius, fs,
                     hrtf_dataset="", hrtf_subject=0,
                     rir_itd_present=False):
    """Apply measured HRTF binaural spatialization (Mode C).

    Currently a placeholder. When a SOFA dataset is provided,
    this will:
    1. Load the HRIR dataset
    2. Find the nearest HRIR direction for source_pos
    3. Convolve each ear's signal with the corresponding HRIR

    Args:
        signal_left: left ear signal
        signal_right: right ear signal
        source_pos: [x, y, z] source position
        left_ear_pos: [x, y, z] left ear position
        right_ear_pos: [x, y, z] right ear position
        head_center: [x, y, z] head center
        head_radius: head radius in meters
        fs: sample rate
        hrtf_dataset: path to SOFA file
        hrtf_subject: subject index in dataset

    Returns:
        (left_out, right_out) 鈥?currently raises NotImplementedError
    """
    # Try to load the dataset
    _load_sofa(hrtf_dataset)

    # Once _load_sofa works, the actual processing would go here:
    # hrir_data, positions, meta = _load_sofa(hrtf_dataset)
    # ... look up nearest HRIR, convolve, return

    return signal_left, signal_right  # unreachable until _load_sofa works
