"""Save simulation results to disk."""

from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.io import wavfile, savemat


def _get_project_root():
    return Path(__file__).resolve().parent.parent.parent


def save_results(stereo_signal, trajectory, scene_name, fs,
                 scene_config=None, results_dir=None,
                 loc_result=None, eval_metrics=None,
                 stereo_raw=None):
    """Save stereo WAV, NPZ metadata, MATLAB .mat file, and localization results.

    Args:
        stereo_signal: (N, 2) stereo samples
        trajectory: (N, 3) source positions
        scene_name: name string
        fs: sample rate
        scene_config: SceneConfig (optional, for .mat export)
        results_dir: output directory
        loc_result: LocalizationResult (optional, v3.0)
        eval_metrics: dict from evaluate() (optional, v3.0)

    Returns:
        (wav_path, npz_path, mat_path) paths to saved files
    """
    if results_dir is None:
        results_dir = _get_project_root() / "results"

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{scene_name}_{timestamp}"

    wav_path = results_dir / f"{base_name}.wav"
    npz_path = results_dir / f"{base_name}.npz"

    # Save WAV (int16 stereo)
    wav_data = (stereo_signal * 32767).astype(np.int16)
    wavfile.write(str(wav_path), fs, wav_data)
    print(f"  WAV saved: {wav_path}")

    # Save NPZ metadata
    np.savez_compressed(
        str(npz_path),
        stereo=stereo_signal,
        trajectory=trajectory,
        fs=fs,
        timestamp=timestamp,
        scene=scene_name,
    )
    print(f"  NPZ saved: {npz_path}")

    mat_path = None
    if scene_config is not None:
        cfg = scene_config
        mat_dict = {
            "stereo_signal": stereo_signal,
            "trajectory": trajectory,
            "fs": fs,
            "scene_name": scene_name,
            "timestamp": timestamp,
            "room_dims": np.array(cfg.room.dimensions),
            "absorption": cfg.room.absorption,
            "max_order": cfg.room.max_order,
            "head_center": np.array(cfg.microphone.head_center),
            "head_radius": cfg.microphone.head_radius,
            "left_ear": np.array(cfg.microphone.left_ear),
            "right_ear": np.array(cfg.microphone.right_ear),
            "source_generator": cfg.source.generator,
            "source_duration": cfg.source.duration,
            "motion_enabled": cfg.motion.enabled,
            "motion_type": cfg.motion.type,
            "hrtf_mode": cfg.microphone.hrtf_mode,
        }

        # Head yaw trajectory for GUI diagnostics
        if hasattr(cfg, '_head_yaw_traj'):
            mat_dict["head_yaw_deg"] = cfg._head_yaw_traj
        # Head-local source azimuth (angle between head facing and source)
        if hasattr(cfg, '_traj_local'):
            from src.localization import compute_truth_doa
            hc = np.array(cfg.microphone.head_center)
            mat_dict["head_src_angle"] = compute_truth_doa(cfg._traj_local, hc)

        # Add raw (pre-HRTF) stereo for waveform comparison
        if stereo_raw is not None:
            mat_dict["stereo_raw"] = stereo_raw

        # Add localization results if available (v3.0)
        if loc_result is not None:
            mat_dict["doa_estimated"] = loc_result.doa_estimated
            if loc_result.doa_itd_only is not None:
                mat_dict["doa_itd_only"] = loc_result.doa_itd_only
            if loc_result.doa_ild_only is not None:
                mat_dict["doa_ild_only"] = loc_result.doa_ild_only
            if loc_result.doa_smoothed is not None:
                mat_dict["doa_smoothed"] = loc_result.doa_smoothed
            mat_dict["loc_timestamps"] = loc_result.timestamps
            mat_dict["loc_method"] = loc_result.method
            mat_dict["lock_achieved"] = loc_result.lock_achieved
            mat_dict["lock_frame"] = loc_result.lock_frame
            # Save localization diagnostics
            if loc_result.itd_per_frame is not None:
                mat_dict["loc_itd"] = loc_result.itd_per_frame
            if loc_result.lateral_angle is not None:
                mat_dict["loc_lateral"] = loc_result.lateral_angle
            if loc_result.phase_mean is not None:
                mat_dict["loc_phase"] = loc_result.phase_mean
            if loc_result.freq_mean is not None:
                mat_dict["loc_freq"] = loc_result.freq_mean
            # Single-frame diagnostic detail
            if loc_result.diag_frame_left is not None:
                mat_dict["diag_frame_l"] = loc_result.diag_frame_left
                mat_dict["diag_frame_r"] = loc_result.diag_frame_right
            if loc_result.diag_freqs is not None:
                mat_dict["diag_freqs"] = loc_result.diag_freqs
            if loc_result.diag_spec_left is not None:
                mat_dict["diag_spec_l"] = loc_result.diag_spec_left
                mat_dict["diag_spec_r"] = loc_result.diag_spec_right
            if loc_result.diag_xphase is not None:
                mat_dict["diag_xcorr"] = loc_result.diag_xphase
            if loc_result.diag_xweight is not None:
                mat_dict["diag_xcorr_lag"] = loc_result.diag_xweight

            # Truth DOA from head-local trajectory (generator-side, not passed to localizer)
            head_center = np.array(cfg.microphone.head_center)
            traj_for_truth = getattr(cfg, '_traj_local', trajectory)
            truth_doa = compute_truth_doa(traj_for_truth, head_center)
            loc_times = loc_result.timestamps
            truth_indices = np.clip(
                (loc_times * fs).astype(int), 0, len(truth_doa) - 1
            )
            mat_dict["doa_truth"] = truth_doa[truth_indices]

        if eval_metrics is not None:
            mat_dict["eval_rmse"] = eval_metrics.get("rmse", 0)
            mat_dict["eval_mae"] = eval_metrics.get("mae", 0)
            mat_dict["eval_accuracy_15deg"] = eval_metrics.get("accuracy_15deg", 0)
            mat_dict["eval_front_back_conf"] = eval_metrics.get("front_back_confusion", 0)
            cm = eval_metrics.get("confusion_matrix", [[0]*4]*4)
            mat_dict["eval_confusion_matrix"] = np.array(cm)

        mat_path = results_dir / f"{base_name}.mat"
        savemat(str(mat_path), mat_dict)
        print(f"  MAT saved: {mat_path}")

    # Save evaluation CSV and JSON
    if loc_result is not None and eval_metrics is not None and scene_config is not None:
        from src.localization import compute_truth_doa
        from src.evaluation.metrics import angular_error
        from src.evaluation.reporting import save_csv, save_evaluation_json

        head_center = np.array(scene_config.microphone.head_center)
        truth_doa = compute_truth_doa(trajectory, head_center)
        loc_times = loc_result.timestamps
        truth_indices = np.clip(
            (loc_times * fs).astype(int), 0, len(truth_doa) - 1
        )
        truth_per_frame = truth_doa[truth_indices]
        errors = angular_error(loc_result.doa_estimated, truth_per_frame)

        csv_path = results_dir / f"{base_name}_loc.csv"
        save_csv(loc_times, loc_result.doa_estimated, truth_per_frame, errors, csv_path)

        json_path = results_dir / f"{base_name}_metrics.json"
        save_evaluation_json(eval_metrics, json_path)

    # Update index
    _update_index(results_dir, base_name, scene_name, timestamp, stereo_signal, fs)

    return str(wav_path), str(npz_path), str(mat_path) if mat_path else None


def _update_index(results_dir, base_name, scene_name, timestamp,
                  stereo_signal, fs):
    """Maintain an index.json of all simulation runs."""
    import json
    index_path = results_dir / "index.json"

    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)
    else:
        index = []

    index.append({
        "id": base_name,
        "scene": scene_name,
        "timestamp": timestamp,
        "duration_s": round(len(stereo_signal) / fs, 2),
        "channels": stereo_signal.shape[1],
        "sample_rate": fs,
    })

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
