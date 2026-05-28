"""Binaural processor dispatcher."""

import numpy as np
from . import hrtf as _analytical

SPEED_OF_SOUND = 343.0


def process_binaural(signal_left, signal_right,
                     source_pos, left_ear_pos, right_ear_pos,
                     head_center, head_radius, fs,
                     mode="analytical",
                     hrtf_dataset="", hrtf_subject=0,
                     head_yaw=0.0,
                     rir_itd_present=False):
    """Apply binaural spatialization with optional head rotation.

    Args:
        signal_left, signal_right: per-ear signals after RIR convolution
        source_pos: [x, y, z] source position (world coords)
        left_ear_pos, right_ear_pos: ear positions
        head_center: head center position
        head_radius: head radius in meters
        fs: sample rate
        mode: "analytical", "parametric", or "measured"
        hrtf_dataset: SOFA file path (measured mode)
        hrtf_subject: subject index (measured mode)
        head_yaw: head yaw angle in radians (+=right)
        rir_itd_present: if True, ITD is already embedded in per-ear RIRs
                         and synthetic ITD is skipped (ILD still applied)

    Returns:
        (left_out, right_out) processed signals
    """
    # Rotate source into head-local coordinates
    src = np.array(source_pos, dtype=np.float64)
    hc = np.array(head_center, dtype=np.float64)
    if abs(head_yaw) > 1e-6:
        rel = src - hc
        cos_y = np.cos(head_yaw); sin_y = np.sin(head_yaw)
        rel_local = np.array([cos_y*rel[0] - sin_y*rel[1],
                               sin_y*rel[0] + cos_y*rel[1], rel[2]])
        src = hc + rel_local

    if mode == "measured":
        from . import hrtf_measured as engine
        return engine.process_binaural(
            signal_left, signal_right,
            source_pos, left_ear_pos, right_ear_pos,
            head_center, head_radius, fs,
            hrtf_dataset=hrtf_dataset, hrtf_subject=hrtf_subject,
            rir_itd_present=rir_itd_present,
        )
    elif mode == "parametric":
        from . import hrtf_parametric as engine
        return engine.process_binaural(
            signal_left, signal_right,
            src, left_ear_pos, right_ear_pos,
            head_center, head_radius, fs,
            rir_itd_present=rir_itd_present,
        )
    else:
        azimuth = _analytical.compute_azimuth(src, head_center)
        azim_abs = abs(azimuth)
        right_is_ipsi = (azimuth >= 0)

        if not rir_itd_present:
            # Synthetic ITD: apply phase-shift delay to contralateral ear
            blend = (np.cos(azimuth - np.pi/2) + 1.0) / 2.0
            left = _analytical.apply_itd(
                signal_left, azim_abs, fs, head_radius, blend=blend)
            right = _analytical.apply_itd(
                signal_right, azim_abs, fs, head_radius, blend=1.0-blend)
        else:
            # ITD already in per-ear RIRs; pass through unchanged
            left = signal_left
            right = signal_right

        # ILD: head shadow broadband attenuation (always applied)
        left = _analytical.apply_ild(
            left, azim_abs, fs, head_radius, not right_is_ipsi)
        right = _analytical.apply_ild(
            right, azim_abs, fs, head_radius, right_is_ipsi)

        return left, right