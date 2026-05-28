"""Main simulation orchestrator."""

from pathlib import Path
import time
import numpy as np
from scipy.signal import fftconvolve

from src.config.loader import load_scene
from src.signals.generator import generate_source
from src.motion.trajectory import generate_trajectory
from src.room.acoustic import compute_rir
from src.spatial.binaural import process_binaural
from src.motion.doppler import apply_doppler
from src.noise import add_noise
from src.localization import create_localizer, compute_truth_doa
from src.evaluation import evaluate
from src.evaluation.metrics import angular_error
from src.evaluation.visualization import (
    plot_doa_trajectory, plot_error_histogram, plot_confusion_matrix
)
from src.evaluation.reporting import save_csv, save_evaluation_json, generate_report


def _determine_segment_count(trajectory, fs, seg_duration=0.05,
                             head_yaw_speed_deg=0.0):
    if len(trajectory) < 2:
        return 1
    dt = 1.0 / fs
    velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1) / dt
    avg_speed = np.mean(velocities)
    wavelength_1khz = 0.343
    max_dist = wavelength_1khz / 4
    seg_time = max_dist / max(avg_speed, 0.01)
    seg_time = max(seg_duration, min(seg_time, 0.2))
    # Head rotation constraint: keep per-segment rotation < 2 deg
    if head_yaw_speed_deg > 1e-6:
        max_seg_for_head = 2.0 / head_yaw_speed_deg
        seg_time = min(seg_time, max(0.01, max_seg_for_head))
    total_time = len(trajectory) / fs
    return max(4, int(total_time / seg_time))


def _overlap_add(source_signal, trajectory, left_ear, right_ear,
                 head_center, head_radius, room_cfg, fs,
                 n_segments, hrtf_mode="analytical", head_yaw=0.0,
                 head_yaw_speed=0.0, active_head=False,
                 head_yaw_traj=None, print_progress=False):
    total_len = len(source_signal)
    segment_len = total_len // n_segments
    hop_len = max(1, segment_len // 2)  # 50% overlap
    n_segments_actual = (total_len - segment_len) // hop_len + 1
    rir_len = min(4096, fs // 20)

    out_left = np.zeros(total_len + rir_len + hop_len)
    out_right = np.zeros(total_len + rir_len + hop_len)
    raw_left = np.zeros(total_len + rir_len + hop_len)
    raw_right = np.zeros(total_len + rir_len + hop_len)

    win = np.sqrt(np.hanning(segment_len + 1)[:segment_len])

    for i in range(n_segments_actual):
        s0 = i * hop_len
        s1 = min(s0 + segment_len, total_len)
        actual_len = s1 - s0

        seg_signal = np.zeros(segment_len)
        seg_signal[:actual_len] = source_signal[s0:s1]
        if actual_len == segment_len:
            seg_win = seg_signal * win
        else:
            w = np.sqrt(np.hanning(actual_len + 1)[:actual_len])
            seg_win = np.pad(seg_signal * w, (0, segment_len - actual_len))

        avg_pos = np.mean(trajectory[s0:s1], axis=0)

        # Time-varying head yaw (precomputed trajectory or simple formula)
        if head_yaw_traj is not None:
            mid = s0 + segment_len // 2
            yaw = np.deg2rad(head_yaw_traj[min(mid, len(head_yaw_traj)-1)])
        else:
            seg_center_s = (s0 + segment_len / 2) / fs
            yaw = head_yaw + np.deg2rad(head_yaw_speed) * seg_center_s

        rir = compute_rir(avg_pos, head_center, room_cfg, fs, rir_len=rir_len)
        conv_l = fftconvolve(seg_win, rir)
        conv_r = fftconvolve(seg_win, rir)
        c_len = min(len(conv_l), len(conv_r))
        conv_l = conv_l[:c_len]
        conv_r = conv_r[:c_len]

        raw_end = min(s0 + len(conv_l), len(raw_left))
        raw_contrib = raw_end - s0
        raw_left[s0:raw_end] += conv_l[:raw_contrib]
        raw_right[s0:raw_end] += conv_r[:raw_contrib]

        seg_l, seg_r = process_binaural(
            conv_l, conv_r,
            avg_pos, left_ear, right_ear,
            head_center, head_radius, fs,
            mode=hrtf_mode,
            head_yaw=yaw,
        )

        offset = s0
        end = min(offset + len(seg_l), len(out_left))
        contrib_len = end - offset
        out_left[offset:end] += seg_l[:contrib_len]
        out_right[offset:end] += seg_r[:contrib_len]

        if print_progress and (i + 1) % 25 == 0:
            print(f"       Segment {i+1}/{n_segments_actual} "
                  f"pos=[{avg_pos[0]:.1f}, {avg_pos[1]:.1f}, {avg_pos[2]:.1f}]")

    trim_len = total_len
    return (out_left[:trim_len], out_right[:trim_len],
            raw_left[:trim_len], raw_right[:trim_len])


class BinauralSimulator:
    def __init__(self, scene_path=None, *, cfg_dict=None, verbose=True):
        if cfg_dict is not None:
            self.cfg = load_scene(cfg_dict)
        elif scene_path is not None:
            self.cfg = load_scene(scene_path)
        else:
            raise ValueError("Either scene_path or cfg_dict must be provided")
        self.verbose = verbose
        self.fs = self.cfg.output.sample_rate
        self.project_root = Path(__file__).resolve().parent.parent.parent

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def run(self):
        cfg = self.cfg
        self._log(f"\n{'='*60}")
        self._log(f"  Binaural Acoustic Simulation v3.0")
        self._log(f"  Scene: {cfg.name}")
        self._log(f"  {cfg.description}")
        self._log(f"{'='*60}\n")

        self._log(f"[1/9] Generating source signal ({cfg.source.generator})...")
        t0 = time.time()
        source_signal = generate_source(cfg.source)
        if cfg.source.sample_rate != self.fs:
            from scipy.signal import resample
            new_len = int(len(source_signal) * self.fs / cfg.source.sample_rate)
            source_signal = resample(source_signal, new_len)
        duration = len(source_signal) / self.fs
        print(f"       Duration: {duration:.2f}s, Samples: {len(source_signal)}")

        print(f"[2/9] Generating trajectory ({cfg.motion.type})...")
        trajectory = generate_trajectory(cfg.motion, duration, self.fs)
        if len(trajectory) > len(source_signal):
            trajectory = trajectory[:len(source_signal)]
        elif len(trajectory) < len(source_signal):
            source_signal = source_signal[:len(trajectory)]
        n_total = len(source_signal)
        print(f"       Points: {n_total}")

        head_yaw_for_seg = 120.0 if cfg.localization.active_head else cfg.motion.head_yaw_speed
        n_segments = _determine_segment_count(trajectory, self.fs,
                                              head_yaw_speed_deg=head_yaw_for_seg)
        seg_ms = (n_total / n_segments) / self.fs * 1000
        self._log(f"[3/9] Segmenting: {n_segments} base segments ({seg_ms:.0f}ms each, 50% overlap)")

        print(f"[4/9] Running room simulation + binaural processing...")
        print(f"       HRTF mode: {cfg.microphone.hrtf_mode}")
        mic = cfg.microphone
        left_ear = np.array(mic.left_ear)
        right_ear = np.array(mic.right_ear)
        head_center = np.array(mic.head_center)

        # --- Helpers for closed-loop head tracking ---
        total_len2 = len(source_signal)

        def _apply_yaw_and_ola(yaw_traj):
            cfg._head_yaw_traj = yaw_traj
            yr = np.deg2rad(yaw_traj)
            cy = np.cos(yr); sy = np.sin(yr)
            rt = trajectory - head_center
            cfg._traj_local = np.column_stack([
                cy * rt[:, 0] - sy * rt[:, 1],
                sy * rt[:, 0] + cy * rt[:, 1],
                rt[:, 2]
            ]) + head_center
            return _overlap_add(source_signal, trajectory,
                left_ear, right_ear, head_center, mic.head_radius,
                cfg.room, self.fs, n_segments,
                hrtf_mode=cfg.microphone.hrtf_mode,
                head_yaw_traj=yaw_traj,
                print_progress=self.verbose)

        def _build_yaw_p1(total_len):
            """Pass 1 yaw: Phase1 still, Phase2 right 15deg, then extrapolate (keep rotating)."""
            yaw = np.zeros(total_len)
            prev = 0.0
            for k in range(total_len):
                t = k / self.fs
                if t < 0.1:
                    y = 0.0
                elif t < 0.35:
                    y = 60.0 * (t - 0.1)   # 15° right at 60°/s
                else:
                    y = prev + 60.0 / self.fs  # extrapolate
                yaw[k] = y
                prev = y
            return yaw

        if cfg.localization.active_head:
            self._log("       Closed-loop: flash-turn-track, localizer-driven")

            # === Pass 1: scanning ===
            yaw_pass1 = _build_yaw_p1(total_len2)
            l1, r1, raw_l1, raw_r1 = _apply_yaw_and_ola(yaw_pass1)
            peak1 = max(np.max(np.abs(l1)), np.max(np.abs(r1)))
            if peak1 > 0: l1, r1 = l1 / peak1 * 0.95, r1 / peak1 * 0.95
            stereo_pass1 = add_noise(np.column_stack([l1, r1]), self.fs, cfg.noise)

            loc_pass1 = create_localizer("gcc_phat", self.fs,
                frame_duration_ms=cfg.localization.frame_duration_ms,
                frame_hop_ms=cfg.localization.frame_hop_ms,
                head_radius=mic.head_radius,
                freq_range=cfg.localization.freq_range,
                active_head=True, verbose=self.verbose)
            res_p1 = loc_pass1.localize(stereo_pass1)
            doa_p1 = res_p1.doa_estimated
            p1_side = loc_pass1._phase1_detect_side(stereo_pass1)
            p2_back = loc_pass1._phase2_detect_fb(stereo_pass1, p1_side, start_frame=4)
            self._log(f"       Pass1: source_left={p1_side} is_back={p2_back}")

            # === Pass 2: flash + closed-loop tracking ===
            fhop = cfg.localization.frame_hop_ms / 1000.0
            flash_amount = 120.0   # degrees
            flash_speed = 480.0    # deg/s
            track_speed = 90.0     # deg/s max tracking
            stop_thresh = 10.0     # |doa| below this → head stops → simulation ends
            settling_time = 0.05   # seconds of stillness after flash
            t_flash_end = 0.35 + flash_amount / flash_speed  # ~0.6s
            t_track_start = t_flash_end + settling_time       # ~0.65s

            yaw_pass2 = np.zeros(total_len2)
            prev = 0.0
            stopped = False
            stop_sample = total_len2  # sample index where simulation ends
            for k in range(total_len2):
                t = k / self.fs
                if t < 0.1:
                    y = 0.0                                    # Phase 1
                elif t < 0.35:
                    y = 60.0 * (t - 0.1)                       # Phase 2: right 15°
                elif p2_back and t < t_flash_end:
                    flash_dir = -1.0 if p1_side else 1.0       # left(-) / right(+)
                    y = prev + flash_dir * flash_speed / self.fs
                elif p2_back and t < t_track_start:
                    y = prev                                    # settling: hold
                elif stopped:
                    y = prev                                    # converged: hold
                else:
                    tidx = int(np.clip(t / fhop, 0, len(doa_p1) - 1))
                    doa_est = doa_p1[tidx]
                    if abs(doa_est) < stop_thresh:
                        stopped = True
                        stop_sample = k
                        y = prev
                    else:
                        world_target = yaw_pass1[k] + doa_est
                        diff = (world_target - prev + 180) % 360 - 180
                        tgt = prev + diff
                        ms = track_speed / self.fs
                        if tgt > prev + ms: y = prev + ms
                        elif tgt < prev - ms: y = prev - ms
                        else: y = tgt
                yaw_pass2[k] = y
                prev = y

            if stopped:
                # Truncate at stop point + small margin for clean segment boundary
                margin = n_segments * 2  # a few OLA segments for smooth tail
                stop_sample = min(stop_sample + margin, total_len2)
                self._log(f"       Converged at t={stop_sample/self.fs:.2f}s, truncating")
                source_signal = source_signal[:stop_sample]
                trajectory = trajectory[:stop_sample]
                yaw_pass2 = yaw_pass2[:stop_sample]
                total_len2 = stop_sample
                n_total = stop_sample
            else:
                self._log(f"       Not converged within signal duration")
            left_out, right_out, raw_left, raw_right = _apply_yaw_and_ola(yaw_pass2)
            head_yaw_traj = yaw_pass2
        else:
            head_yaw_traj = np.zeros(total_len2)
            head_yaw_traj[:] = cfg.microphone.head_yaw_deg + cfg.motion.head_yaw_speed * np.arange(total_len2) / self.fs
            # Compute traj_local for passive mode
            yr = np.deg2rad(head_yaw_traj)
            cy = np.cos(yr); sy = np.sin(yr)
            rt = trajectory - head_center
            cfg._traj_local = np.column_stack([
                cy * rt[:, 0] - sy * rt[:, 1],
                sy * rt[:, 0] + cy * rt[:, 1],
                rt[:, 2]
            ]) + head_center
            cfg._head_yaw_traj = head_yaw_traj
            left_out, right_out, raw_left, raw_right = _overlap_add(
                source_signal, trajectory,
                left_ear, right_ear, head_center, mic.head_radius,
                cfg.room, self.fs, n_segments,
                hrtf_mode=cfg.microphone.hrtf_mode,
                head_yaw_traj=head_yaw_traj,
                print_progress=self.verbose,
            )

        print(f"[5/9] Applying Doppler effect...")
        left_out = apply_doppler(left_out, trajectory, left_ear, self.fs)
        right_out = apply_doppler(right_out, trajectory, right_ear, self.fs)

        print(f"[6/9] Normalizing...")
        peak = max(np.max(np.abs(left_out)), np.max(np.abs(right_out)))
        if peak > 0:
            left_out = left_out / peak * 0.95
            right_out = right_out / peak * 0.95
        raw_peak = max(np.max(np.abs(raw_left)), np.max(np.abs(raw_right)))
        if raw_peak > 0:
            raw_left = raw_left / raw_peak * 0.95
            raw_right = raw_right / raw_peak * 0.95

        stereo = np.column_stack([left_out, right_out])
        stereo_raw = np.column_stack([raw_left, raw_right])

        print(f"[7/9] Adding noise (background={cfg.noise.background_snr_db}dB, "
              f"sensor={cfg.noise.sensor_snr_db}dB)...")
        noisy_stereo = add_noise(stereo, self.fs, cfg.noise)

        loc_result = None; eval_metrics = None
        if cfg.evaluation.enabled:
            print(f"[8/9] Running localization ({cfg.localization.method})...")
            loc = create_localizer(
                cfg.localization.method, self.fs,
                frame_duration_ms=cfg.localization.frame_duration_ms,
                frame_hop_ms=cfg.localization.frame_hop_ms,
                head_radius=cfg.microphone.head_radius,
                freq_range=cfg.localization.freq_range,
                head_yaw_speed=cfg.localization.head_yaw_speed,
                active_head=cfg.localization.active_head,
            )
            try:
                loc_result = loc.localize(noisy_stereo)
            except TypeError:
                loc_result = loc.localize(noisy_stereo, stereo_raw=stereo_raw)
            print(f"       Estimated {loc_result.doa_estimated.shape[0]} frames")

            print(f"[9/9] Evaluating localization accuracy...")
            truth_doa = compute_truth_doa(cfg._traj_local, head_center)
            loc_times = loc_result.timestamps
            truth_indices = np.clip(
                (loc_times * self.fs).astype(int), 0, len(truth_doa) - 1
            )
            truth_per_frame = truth_doa[truth_indices]
            eval_metrics = evaluate(
                loc_result.doa_estimated, truth_per_frame,
                timestamps=loc_times, scene_name=cfg.name
            )
            print(f"       RMSE: {eval_metrics['rmse']:.2f} deg")
            print(f"       15 deg Accuracy: {eval_metrics['accuracy_15deg']*100:.1f}%")
            print(f"       Front-Back Conf: {eval_metrics['front_back_confusion']*100:.1f}%")
        else:
            print(f"[8/9] Localization skipped (evaluation disabled)")
            print(f"[9/9] Evaluation skipped")

        elapsed = time.time() - t0
        print(f"\n  Simulation complete in {elapsed:.1f}s")
        print(f"  Output: {stereo.shape} samples, "
              f"{stereo.shape[0]/self.fs:.2f}s stereo")

        return stereo, trajectory, loc_result, eval_metrics, stereo_raw
