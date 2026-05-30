"""Offset controller + state machine for continuous source tracking.

States: SEARCH -> TRACKING -> STEADY -> STOP -> SEARCH (cycle)

v3.0: Time-delay offset replaces fixed geometric offset.
      Head chases theta_src(t - tau) where tau adapts to source speed.
      Fast source -> short tau (larger gap), slow source -> long tau.
      Static source -> tau converges, gap = 0 (no offset needed).
"""

import enum
import numpy as np


class TrackingState(enum.Enum):
    SEARCH = "search"       # coarse localisation, head rotating
    TRACKING = "tracking"   # Kalman active, head follows delayed theta_src
    STEADY = "steady"       # source nearly static, head aligned (no offset)
    STOP = "stop"           # signal lost, head holds position


class OffsetController:
    """State machine with adaptive time-delay offset.

    Core formula:
        tau = max(tau_min, theta_min / max(|omega_src|, epsilon))
        yaw_target = theta_src(t - tau)

    This guarantees:
        - gap >= theta_min whenever source moves
        - gap = 0 when source is static (head aligns naturally)
        - no direction judgment, no alternating logic

    Usage::

        ctrl = OffsetController(theta_min=5.0, tau_min=0.5)
        theta_src, omega_src = kalman.update(...)
        yaw_target = ctrl.get_yaw_target(theta_src, omega_src)
        state = ctrl.state_transition(...)
    """

    def __init__(self, theta_min: float = 5.0,
                 tau_min: float = 0.5,
                 max_history_s: float = 10.0,
                 search_yaw_speed: float = 120.0,
                 tracking_yaw_speed: float = 90.0,
                 signal_timeout_s: float = 1.0):
        # ---- Time-delay offset parameters ----
        self.theta_min = theta_min        # minimum angular gap (deg)
        self.tau_min = tau_min            # minimum lookback (seconds)
        self.tau_max = 2.0                # maximum lookback (seconds)
        self.omega_epsilon = 0.01         # deg/s floor to prevent div-by-zero

        # ---- Ring buffer for theta_src history ----
        self._max_history_s = max_history_s
        self._history_len = 200  # will be resized on first frame
        self._theta_history = np.zeros(self._history_len)
        self._hist_ptr = 0               # write pointer (most recent)
        self._hist_count = 0             # total frames written

        # ---- Speed limits ----
        self.search_yaw_speed = search_yaw_speed
        self.tracking_yaw_speed = tracking_yaw_speed

        # ---- State ----
        self.state = TrackingState.SEARCH
        self._state_timer = 0.0
        self._silence_timer = 0.0
        self._last_theta_src = 0.0

        # ---- Thresholds ----
        self.convergence_threshold = 30.0
        self.covariance_stable_threshold = 5.0
        self.omega_stable_threshold = 10.0    # |omega| below this -> steady
        self.omega_exit_steady = 2.0          # |omega| above this -> exit STEADY
        self.signal_energy_threshold = 1e-6
        self.timeout_s = signal_timeout_s
        self.residual_divergence = 30.0

        self._dt = 0.025

        # ---- Computed values (readonly) ----
        self._current_tau = 0.0
        self._current_gap = 0.0

    # ==========  Time-Delay Offset  ==========

    def get_yaw_target(self, theta_src: float, omega_src: float = 0.0) -> float:
        """Compute yaw target with adaptive time-delay offset.

        Hybrid strategy:
        - Buffer not full (< tau_min history): fixed geometric offset (?theta_min)
        - Buffer full: time-delayed theta_src
        - Source nearly static (|omega| < 1 deg/s): align with source (no offset)
        """
        # Ensure buffer is sized
        needed = int(self._max_history_s / self._dt)
        if needed > self._history_len:
            self._history_len = needed
            new_buf = np.zeros(self._history_len)
            copy_len = min(self._hist_count, len(new_buf))
            new_buf[:copy_len] = self._theta_history[:copy_len]
            self._theta_history = new_buf

        # Write current into ring buffer (normalized to [-180, 180])
        theta_normalized = ((theta_src + 180) % 360) - 180
        self._theta_history[self._hist_ptr] = theta_normalized
        self._hist_ptr = (self._hist_ptr + 1) % self._history_len
        self._hist_count = min(self._hist_count + 1, self._history_len)
        self._last_theta_src = theta_src

        abs_omega = max(abs(omega_src), self.omega_epsilon)

        # ---- Static / near-static: align with source ----
        if abs_omega < 1.0 and self._hist_count >= int(self.tau_min / self._dt):
            self._current_tau = 0.0
            self._current_gap = 0.0
            target = theta_src
        # ---- Buffer not full: geometric offset to keep ITD alive ----
        elif self._hist_count < int(self.tau_min / self._dt):
            self._current_tau = 0.0
            self._current_gap = self.theta_min
            # Offset opposite to motion direction (or source side if static)
            if abs(omega_src) > 1.0:
                sign = 1 if omega_src > 0 else -1
            else:
                sign = 1 if theta_normalized > 0 else -1
            target = theta_src - sign * self.theta_min
        # ---- Moving, buffer full: time-delay ----
        else:
            raw_tau = self.theta_min / abs_omega
            self._current_tau = max(self.tau_min, min(raw_tau, self.tau_max))
            self._current_gap = abs_omega * self._current_tau

            delay_frames = int(self._current_tau / self._dt)
            delay_frames = min(delay_frames, self._hist_count - 1)
            delay_frames = max(delay_frames, 0)

            if delay_frames > 0:
                read_idx = (self._hist_ptr - 1 - delay_frames) % self._history_len
                target = float(self._theta_history[read_idx])
            else:
                target = theta_src

        # Wrap
        target = ((target + 180) % 360) - 180
        return target


    @property
    def current_tau(self) -> float:
        return self._current_tau

    @property
    def current_gap(self) -> float:
        return self._current_gap

    # ==========  State Machine  ==========

    def state_transition(self, theta_src: float, omega_src: float,
                         covariance: float, signal_energy: float,
                         frame_hop_ms: float = 25.0):
        """Evaluate state transition. Call once per frame.

        Returns:
            new_state (TrackingState)
        """
        self._dt = frame_hop_ms / 1000.0
        self._state_timer += self._dt

        # Signal detection
        has_signal = signal_energy > self.signal_energy_threshold
        if has_signal:
            self._silence_timer = 0.0
        else:
            self._silence_timer += self._dt

        # Global: signal loss -> STOP
        if self._silence_timer > self.timeout_s:
            if self.state != TrackingState.STOP:
                self.state = TrackingState.STOP
            return self.state

        # Global: signal returns -> SEARCH
        if self.state == TrackingState.STOP and has_signal:
            self.state = TrackingState.SEARCH
            self._state_timer = 0.0
            # Reset buffer
            self._hist_ptr = 0
            self._hist_count = 0

        if self.state == TrackingState.SEARCH:
            if self._state_timer > 0.5 and abs(theta_src) < self.convergence_threshold:
                self.state = TrackingState.TRACKING
                self._state_timer = 0.0

        elif self.state == TrackingState.TRACKING:
            omega_ok = abs(omega_src) < self.omega_stable_threshold
            cov_ok = covariance < self.covariance_stable_threshold
            if omega_ok and cov_ok and self._state_timer > 1.0:
                self.state = TrackingState.STEADY
                self._state_timer = 0.0

        elif self.state == TrackingState.STEADY:
            # Exit STEADY when source moves again (no perturbation needed)
            if abs(omega_src) > self.omega_exit_steady:
                self.state = TrackingState.TRACKING
                self._state_timer = 0.0

        return self.state
