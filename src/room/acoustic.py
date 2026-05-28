"""Room acoustics simulation using pyroomacoustics image-source method."""

import numpy as np
import pyroomacoustics as pra

from src.config.schema import RoomConfig


def compute_rir(source_pos, mic_pos, room_cfg: RoomConfig, fs: int,
                rir_len: int = None) -> np.ndarray:
    """Compute room impulse response from source to microphone.

    Args:
        source_pos: [x, y, z] source position
        mic_pos: [x, y, z] microphone position
        room_cfg: RoomConfig with dimensions, absorption, max_order
        fs: sample rate in Hz
        rir_len: RIR length in samples (default: fs//20 ~ 50ms)

    Returns:
        1-D numpy array: room impulse response
    """
    if rir_len is None:
        rir_len = fs // 20  # ~50ms

    absorption = room_cfg.absorption
    max_order = room_cfg.max_order

    room = pra.ShoeBox(
        room_cfg.dimensions,
        fs=fs,
        absorption=absorption,
        max_order=max_order,
    )

    impulse = np.zeros(rir_len)
    impulse[0] = 1.0
    room.add_source(source_pos, signal=impulse)

    mic_array = np.array([[mic_pos[0]], [mic_pos[1]], [mic_pos[2]]])
    room.add_microphone_array(mic_array)

    room.simulate()
    rir = room.mic_array.signals[0, :]

    if len(rir) < rir_len:
        rir = np.pad(rir, (0, rir_len - len(rir)))
    else:
        rir = rir[:rir_len]

    return rir


def compute_stereo_rir(source_pos, left_ear, right_ear,
                       room_cfg: RoomConfig, fs: int, rir_len: int = None):
    """Compute RIR for both ears simultaneously using one ShoeBox simulation.

    Args:
        source_pos: [x, y, z] source position
        left_ear: [x, y, z] left ear position
        right_ear: [x, y, z] right ear position
        room_cfg: RoomConfig with dimensions, absorption, max_order
        fs: sample rate in Hz
        rir_len: RIR length in samples (default: fs//10 ~ 100ms)

    Returns:
        (rir_left, rir_right) as 1-D arrays
    """
    if rir_len is None:
        rir_len = fs // 10  # ~100ms

    absorption = room_cfg.absorption
    max_order = room_cfg.max_order

    room = pra.ShoeBox(
        room_cfg.dimensions,
        fs=fs,
        absorption=absorption,
        max_order=max_order,
    )

    impulse = np.zeros(rir_len)
    impulse[0] = 1.0
    room.add_source(source_pos, signal=impulse)

    # Stereo microphone array at ear positions
    mic_array = np.array([
        [left_ear[0], right_ear[0]],
        [left_ear[1], right_ear[1]],
        [left_ear[2], right_ear[2]],
    ])
    room.add_microphone_array(mic_array)

    room.simulate()
    rir_l = room.mic_array.signals[0, :]
    rir_r = room.mic_array.signals[1, :]

    # Pad/truncate to requested length
    if len(rir_l) < rir_len:
        rir_l = np.pad(rir_l, (0, rir_len - len(rir_l)))
    else:
        rir_l = rir_l[:rir_len]
    if len(rir_r) < rir_len:
        rir_r = np.pad(rir_r, (0, rir_len - len(rir_r)))
    else:
        rir_r = rir_r[:rir_len]

    return rir_l, rir_r