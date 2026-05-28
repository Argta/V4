fpath = r"C:\Users\LENOVO\Documents\shuanger\src\pipeline\simulator.py"
with open(fpath, 'rb') as f:
    raw = f.read()

# Find old active_head block: from "if cfg.localization.active_head:" 
# to the passive "else:" or similar boundary
start = raw.find(b'if cfg.localization.active_head:')
# Find the matching passive else: that starts with "head_yaw_traj = np.zeros"
else_marker = b"head_yaw_traj = np.zeros(total_len2)"
end = raw.find(else_marker, start)
# Go back to include the "else:" line
end = raw.rfind(b'else:', start, end) - 1  # back to newline before else

print(f'Replacing bytes {start} to {end} ({end-start} bytes)')

new_code = b'''        if cfg.localization.active_head:
            self._log("       LLR-driven active head (v4.0)")

            # === Pass 1: scanning with LLR ===
            fhop = cfg.localization.frame_hop_ms / 1000.0
            yaw_pass1 = _build_yaw_p1(total_len2)
            l1, r1, raw_l1, raw_r1 = _apply_yaw_and_ola(yaw_pass1)
            peak1 = max(np.max(np.abs(l1)), np.max(np.abs(r1)))
            if peak1 > 0: l1, r1 = l1 / peak1 * 0.95, r1 / peak1 * 0.95
            stereo_pass1 = add_noise(np.column_stack([l1, r1]), self.fs, cfg.noise)

            llr_thresh = getattr(cfg.localization, 'llr_base_threshold', 3.0)
            stag_timeout = getattr(cfg.localization, 'stagnation_timeout_ms', 500.0)
            loc_llr = create_localizer(
                "llr", self.fs,
                frame_duration_ms=cfg.localization.frame_duration_ms,
                frame_hop_ms=cfg.localization.frame_hop_ms,
                head_radius=mic.head_radius,
                freq_range=cfg.localization.freq_range,
                head_yaw_speed=60.0,
                llr_base_threshold=llr_thresh,
                stagnation_timeout_ms=stag_timeout,
                verbose=self.verbose)
            res_p1 = loc_llr.localize(stereo_pass1)
            doa_p1 = res_p1.doa_estimated
            self._log(f"       LLR: fb_determined={res_p1.fb_determined} is_back={res_p1.fb_is_back}")

            # === Handle stagnation: perturbation ===
            if res_p1.action_suggestion == "perturb_rotation" and not res_p1.fb_determined:
                self._log("       Stagnation detected: applying perturbation")
                perturb_yaw = _build_yaw_with_perturb(total_len2, self.fs, 0.35)
                l1, r1, _, _ = _apply_yaw_and_ola(perturb_yaw)
                peak1 = max(np.max(np.abs(l1)), np.max(np.abs(r1)))
                if peak1 > 0: l1, r1 = l1 / peak1 * 0.95, r1 / peak1 * 0.95
                stereo_p = add_noise(np.column_stack([l1, r1]), self.fs, cfg.noise)
                res_p1 = loc_llr.localize(stereo_p)
                doa_p1 = res_p1.doa_estimated
                yaw_pass1 = perturb_yaw

            # === Pass 2: tracking with FB knowledge ===
            track_speed = 90.0
            stop_thresh = 10.0

            yaw_pass2 = np.zeros(total_len2)
            prev = 0.0
            stopped = False
            stop_sample = total_len2
            for k in range(total_len2):
                t = k / self.fs
                if t < 0.1:
                    y = 0.0
                elif t < 0.35:
                    y = 60.0 * (t - 0.1)
                elif stopped:
                    y = prev
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
                margin = n_segments * 2
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
        '''

new_raw = raw[:start] + new_code + raw[end:]
with open(fpath, 'wb') as f:
    f.write(new_raw)
print(f'Done. New size: {len(new_raw)}')
