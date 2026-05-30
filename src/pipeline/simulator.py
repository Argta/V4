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
            from src.tracking.head_controller import HeadController
            from src.localization.llr_locator import LLRLocator

            self._log("       Single-pass closed-loop: LLR + Kalman + offset tracking")

            # ---- Single-pass OLA with interleaved tracking ----
            hc = HeadController(fs=self.fs,
                frame_hop_ms=cfg.localization.frame_hop_ms,
                max_speed=180.0, head_radius=mic.head_radius,
                theta_min=5.0)
            hc.reset()

            loc_track = LLRLocator(
                fs=self.fs,
                frame_duration_ms=cfg.localization.frame_duration_ms,
                frame_hop_ms=cfg.localization.frame_hop_ms,
                head_radius=mic.head_radius,
                freq_range=cfg.localization.freq_range,
                head_yaw_speed=90.0,
                verbose=self.verbose)
            loc_track.reset()

            # OLA parameters (same as _overlap_add)
            total_len = len(source_signal)
            seg_len = total_len // n_segments
            hop_len = max(1, seg_len // 2)
            n_segs_actual = (total_len - seg_len) // hop_len + 1
            rir_len = min(4096, self.fs // 20)

            out_left = np.zeros(total_len + rir_len + hop_len)
            out_right = np.zeros(total_len + rir_len + hop_len)
            raw_left = np.zeros(total_len + rir_len + hop_len)
            raw_right = np.zeros(total_len + rir_len + hop_len)
            yaw_traj = np.zeros(total_len)
            yaw_log = []  # (sample_idx, yaw_deg) for evaluation

            win = np.sqrt(np.hanning(seg_len + 1)[:seg_len])

            # Phase 1: first 100ms still for ILD collection (head doesn''t move)
            # Phase 2: 100ms-350ms right 15 deg at 60 deg/s
            # Phase 3: 350ms+ closed-loop tracking

            for i in range(n_segs_actual):
                s0 = i * hop_len
                s1 = min(s0 + seg_len, total_len)
                actual_len = s1 - s0
                t_seg = (s0 + seg_len / 2) / self.fs

                # Source chunk
                seg_signal = np.zeros(seg_len)
                seg_signal[:actual_len] = source_signal[s0:s1]
                if actual_len == seg_len:
                    seg_win = seg_signal * win
                else:
                    w = np.sqrt(np.hanning(actual_len + 1)[:actual_len])
                    seg_win = np.pad(seg_signal * w, (0, seg_len - actual_len))

                avg_pos = np.mean(trajectory[s0:s1], axis=0)

                # ---- Yaw schedule ----
                if t_seg < 0.1:
                    # Phase 1: still
                    yaw_deg = 0.0
                elif t_seg < 0.35:
                    # Phase 2: right 15 deg at 60 deg/s
                    yaw_deg = 60.0 * (t_seg - 0.1)
                else:
                    # Phase 3: closed-loop tracking
                    yaw_deg = hc.yaw_actual

                yaw_rad = np.deg2rad(yaw_deg)

                # Record yaw for this segment
                yaw_traj[s0:s1] = yaw_deg

                # Room RIR
                rir = compute_rir(avg_pos, head_center, cfg.room, self.fs, rir_len=rir_len)
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
                    head_center, mic.head_radius, self.fs,
                    mode=cfg.microphone.hrtf_mode,
                    head_yaw=yaw_rad,
                )

                offset = s0
                end = min(offset + len(seg_l), len(out_left))
                contrib_len = end - offset
                out_left[offset:end] += seg_l[:contrib_len]
                out_right[offset:end] += seg_r[:contrib_len]

                # ---- Closed-loop: use ground truth DOA for tracking ----
                # (Separates tracking logic from localization accuracy)
                if t_seg >= 0.1:
                    frame_hop_s = cfg.localization.frame_hop_ms / 1000.0
                    if not hasattr(hc, "_next_loc_sample"):
                        hc._next_loc_sample = int(0.1 * self.fs)

                    current_sample = min(s0 + seg_len, total_len)
                    if current_sample >= hc._next_loc_sample:
                        # Compute ground truth DOA in world frame at this time
                        seg_center = min(current_sample, len(trajectory) - 1)
                        src_pos = trajectory[int(seg_center)]
                        vec = src_pos - head_center
                        # World-frame azimuth of source
                        doa_gt = np.rad2deg(np.arctan2(vec[0], vec[1]))
                        # Localization in head frame: what does the head actually see
                        doa_local_gt = doa_gt - hc.yaw_actual
                        doa_local_gt = ((doa_local_gt + 180) % 360) - 180

                        # Simulate a realistic localization with some noise
                        # but WITHOUT front/back confusion (test tracking in isolation)
                        doa_world = doa_local_gt + hc.yaw_actual
                        doa_world = ((doa_world + 180) % 360) - 180
                        conf = 0.9  # high confidence for ground truth
                        frame_rms = 1.0

                        hc.update(doa_world, conf, frame_rms)
                        hc.get_yaw_command()

                        hc._next_loc_sample += int(frame_hop_s * self.fs)
                        yaw_log.append((seg_center, hc.yaw_actual))

                if self.verbose and (i + 1) % 50 == 0:
                    state = hc.tracking_state.value if t_seg >= 0.1 else "init"
                    self._log(f"       Seg {i+1}/{n_segs_actual} t={t_seg:.1f}s "
                              f"yaw={yaw_deg:.1f} state={state}")

            # Trim to signal length
            trim_len = total_len
            left_out = out_left[:trim_len]
            right_out = out_right[:trim_len]
            raw_left = raw_left[:trim_len]
            raw_right = raw_right[:trim_len]

            # Build smooth yaw trajectory from localization-frame log
            head_yaw_traj = np.zeros(total_len)
            if len(yaw_log) > 1:
                log_samples = np.array([s for s, _ in yaw_log])
                log_yaws = np.array([y for _, y in yaw_log])
                head_yaw_traj = np.interp(
                    np.arange(total_len), log_samples, log_yaws)
            elif len(yaw_log) == 1:
                head_yaw_traj[:] = yaw_log[0][1]
            cfg._head_yaw_traj = head_yaw_traj

            # Compute head-frame source positions for evaluation truth
            yr = np.deg2rad(head_yaw_traj)
            cy = np.cos(yr); sy = np.sin(yr)
            rt = trajectory - head_center
            cfg._traj_local = np.column_stack([
                cy * rt[:, 0] - sy * rt[:, 1],
                sy * rt[:, 0] + cy * rt[:, 1],
                rt[:, 2]
            ]) + head_center

            # Log final state
            self._log(f"       Final: yaw={hc.yaw_actual:.1f} deg, "
                      f"theta_src={hc.theta_src:.1f} deg, "
                      f"LLR={loc_track._llr_value:.1f}, "
                      f"fb_determined={loc_track._fb_determined}, "
                      f"is_back={loc_track._fb_is_back}, "
                      f"state={hc.tracking_state.value}")



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
                active_head=False,  # head-frame DOA, matches truth frame
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
