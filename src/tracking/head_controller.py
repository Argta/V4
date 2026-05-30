"""Head controller: PID servo + dual-mode arbitration + speed constraint.

Central orchestrator sitting between the localization system and the
physical/virtual head motor. Combines Kalman tracking, offset policy,
and manual wheel override into a single control loop.

Modes:
    OFFSET_TRACKING (default) - head follows theta_src + offset via Kalman
    WHEEL                    - user manually controls head via GUI slider
"""

import enum
import numpy as np
from .kalman_tracker import KalmanBinauralTracker
from .offset_controller import OffsetController, TrackingState


class ControlMode(enum.Enum):
    OFFSET_TRACKING = "offset_tracking"
    WHEEL = "wheel"


class HeadController:
    """Head motion controller with PID servo and dual-mode arbitration.

    Usage::

        hc = HeadController(fs=48000, frame_hop_ms=25, max_speed=60.0)
        hc.reset()
        for frame in stream:
            doa_world, conf = loc.process_frame(frame, hc.yaw_actual)
            # doa_world is world-frame azimuth
            hc.update(doa_world, conf, signal_rms)
            yaw_target = hc.get_yaw_command()
            # Apply yaw_target to motor/simulator

        # Manual override via GUI wheel:
        hc.wheel_start()
        hc.wheel_delta(delta_deg)  # called repeatedly while dragging
        hc.wheel_stop()            # release back to offset tracking
    """

    def __init__(self, fs: int = 48000, frame_hop_ms: float = 25.0,
                 max_speed: float = 120.0, head_radius: float = 0.09,
                 theta_min: float = 5.0):
        self.fs = fs
        self.dt = frame_hop_ms / 1000.0
        self.max_speed = max_speed      # deg/s maximum head rotation rate

        # Sub-modules
        self.kalman = KalmanBinauralTracker(
            fs=fs, frame_hop_ms=frame_hop_ms, head_radius=head_radius)
        self.offset_ctrl = OffsetController(
            theta_min=theta_min,
            tracking_yaw_speed=max_speed)

        # PID gains (tuned for time-delay tracking)
        self.Kp = 4.0    # proportional gain (higher for faster response)
        self.Ki = 0.3    # integral gain (eliminates steady-state lag)
        self.Kd = 0.15   # derivative gain (low to avoid oscillation)

        # State
        self.mode = ControlMode.OFFSET_TRACKING
        self.yaw_actual = 0.0          # current head yaw (deg)
        self.yaw_target = 0.0          # desired head yaw (deg)
        self._prev_error = 0.0
        self._error_integral = 0.0
        self._theta_src = 0.0          # Kalman-estimated source azimuth
        self._omega_src = 0.0          # Kalman-estimated source angular velocity
        self._confidence = 0.0
        self._signal_energy = 0.0

        # Wheel state
        self._wheel_yaw_at_touch = 0.0  # yaw_actual at takeover moment
        self._wheel_delta_accum = 0.0   # accumulated delta from wheel
        self._transition_remaining = 0.0  # seconds remaining in transition
        self._transition_target = 0.0   # target yaw after transition
        self._transition_duration = 0.3  # 300ms transition

    def reset(self):
        """Reset controller for a new session."""
        self.kalman.reset()
        self.offset_ctrl = OffsetController(
            theta_min=self.offset_ctrl.theta_min,
            tracking_yaw_speed=self.max_speed)
        self.mode = ControlMode.OFFSET_TRACKING
        self.yaw_actual = 0.0
        self.yaw_target = 0.0
        self._prev_error = 0.0
        self._error_integral = 0.0
        self._theta_src = 0.0
        self._omega_src = 0.0
        self._confidence = 0.0
        self._signal_energy = 0.0
        self._wheel_yaw_at_touch = 0.0
        self._wheel_delta_accum = 0.0
        self._transition_remaining = 0.0

    # ---- Main update (call once per frame) ----

    def update(self, doa_world: float, confidence: float,
               signal_rms: float = 1.0):
        """Consume one frame of localization output.

        Args:
            doa_world: world-frame DOA estimate from localizer
            confidence: 0-1 confidence from localizer
            signal_rms: frame RMS energy (for silence detection)
        """
        self._confidence = confidence
        self._signal_energy = signal_rms

        # Kalman update: observation = doa_world (which is theta_src estimate)
        # Note: doa_world = theta_local + yaw_actual, so it directly estimates
        # theta_src (modulo the offset, which the Kalman doesn't know about).
        # We feed the raw world-frame DOA as observation.
        self._theta_src, self._omega_src, P_diag = self.kalman.update(
            doa_world, confidence)

        # Offset controller state transition
        cov = self.kalman.covariance
        self.offset_ctrl.state_transition(
            self._theta_src, self._omega_src, cov,
            signal_rms,
            frame_hop_ms=self.dt * 1000.0)

        # Compute target based on mode
        if self.mode == ControlMode.OFFSET_TRACKING:
            self._update_offset_tracking_target()
        elif self.mode == ControlMode.WHEEL:
            # Target set by wheel_start / wheel_delta
            pass
        # Transition (wheel release) is handled by _update_offset_tracking_target

    def _update_offset_tracking_target(self):
        """Set yaw_target from offset controller (or transition)."""
        if self._transition_remaining > 0:
            # In release transition: blend toward offset target
            offset_target = self.offset_ctrl.get_yaw_target(self._theta_src, self._omega_src)
            self._transition_remaining -= self.dt
            if self._transition_remaining <= 0:
                self.yaw_target = offset_target
                self.mode = ControlMode.OFFSET_TRACKING
            # else: target stays at _transition_target (fixed during transition)
        else:
            self.yaw_target = self.offset_ctrl.get_yaw_target(self._theta_src, self._omega_src)

    # ---- PID servo (call after update, returns the yaw to apply) ----

    def get_yaw_command(self) -> float:
        """Compute PID output and advance yaw_actual by one frame.

        Returns:
            yaw_actual after this frame (deg)
        """
        # Angle normalization
        error = self._normalize_angle(self.yaw_target - self.yaw_actual)

        # PID
        p_term = self.Kp * error
        self._error_integral += error * self.dt
        # Clamp integral to prevent windup
        self._error_integral = np.clip(self._error_integral, -60.0, 60.0)
        i_term = self.Ki * self._error_integral
        d_term = self.Kd * (error - self._prev_error) / self.dt
        output = p_term + i_term + d_term

        # Clamp to max speed
        max_step = self.max_speed * self.dt
        output = np.clip(output, -max_step, max_step)

        # Apply
        self.yaw_actual += output
        self.yaw_actual = self._normalize_angle(self.yaw_actual)
        self._prev_error = error

        return self.yaw_actual

    # ---- Wheel manual control ----

    def wheel_start(self):
        """User touches the wheel - takeover to WHEEL mode."""
        self._wheel_yaw_at_touch = self.yaw_actual
        self._wheel_delta_accum = 0.0
        self.mode = ControlMode.WHEEL
        self.yaw_target = self.yaw_actual

    def wheel_delta(self, delta_deg: float):
        """User drags the wheel by delta_deg.

        Args:
            delta_deg: incremental change from wheel (positive = right)
        """
        if self.mode != ControlMode.WHEEL:
            return
        self._wheel_delta_accum += delta_deg
        self.yaw_target = self._wheel_yaw_at_touch + self._wheel_delta_accum
        # Wrap
        self.yaw_target = ((self.yaw_target + 180) % 360) - 180

    def wheel_stop(self):
        """User releases the wheel - transition back to OFFSET_TRACKING."""
        if self.mode != ControlMode.WHEEL:
            return
        # Get Kalman estimate (use latest even during wheel mode)
        # Kalman has been updating with doa_world observations
        # Predict a bit ahead for the transition
        theta_pred, _ = self.kalman.predict(self._transition_duration)
        offset_target = self.offset_ctrl.get_yaw_target(theta_pred, self._omega_src)

        self._transition_target = offset_target
        self._transition_remaining = self._transition_duration
        self.yaw_target = self._transition_target
        # Mode stays WHEEL until transition completes in
        # _update_offset_tracking_target / mode or is handled in update:
        self.mode = ControlMode.OFFSET_TRACKING

        # Clear wheel state
        self._wheel_delta_accum = 0.0

    # ---- Properties ----

    @property
    def theta_src(self) -> float:
        return self._theta_src

    @property
    def omega_src(self) -> float:
        return self._omega_src

    @property
    def tracking_state(self) -> TrackingState:
        return self.offset_ctrl.state

    @property
    def is_in_transition(self) -> bool:
        return self._transition_remaining > 0

    # ---- Helpers ----

    @staticmethod
    def _normalize_angle(a: float) -> float:
        """Wrap angle to [-180, 180]."""
        return ((a + 180) % 360) - 180
