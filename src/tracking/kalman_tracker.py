"""Kalman filter for binaural source tracking.

State: [theta_src (deg), omega_src (deg/s)]
Observation: theta_src = theta_local + yaw - offset

Eliminates per-frame GCC-PHAT jitter and provides short-term
prediction for offset-direction transitions.
"""

import numpy as np


class KalmanBinauralTracker:
    """2-state constant-velocity Kalman filter for source azimuth tracking.

    Usage::

        kf = KalmanBinauralTracker(fs=48000, frame_hop_ms=25)
        kf.reset()
        for frame in stream:
            doa_local, conf = loc.process_frame(frame)
            yaw_eff = yaw_current - offset_current
            theta_src, omega_src, P = kf.update(doa_local + yaw_eff, conf)
            theta_pred, omega_pred = kf.predict(0.1)  # 100ms ahead
    """

    def __init__(self, fs: int = 48000, frame_hop_ms: float = 25.0,
                 head_radius: float = 0.09):
        self.fs = fs
        self.dt = frame_hop_ms / 1000.0  # seconds per frame

        # State: [theta_src (deg), omega_src (deg/s)]
        self.x = np.zeros(2)
        self.P = np.eye(2) * 100.0  # initial uncertainty

        # State transition: constant velocity
        self.F = np.array([[1, self.dt],
                           [0, 1]])

        # Observation matrix: we observe theta directly
        self.H = np.array([[1, 0]])

        # Default noise parameters
        self.Q_base = np.array([[0.5, 0],
                                [0, 5.0]])  # base process noise
        self.R_base = 25.0              # base measurement noise (deg^2)

        # Adaptive thresholds
        self.residual_threshold = 20.0   # deg - above this ? high-maneuver
        self.Q_boost_factor = 50.0       # multiply Q when maneuvering
        self.adaption_decay = 0.95       # decay factor for boosted Q

        self._boost_active = False
        self.Q_current = self.Q_base.copy()
        self._initialized = False

    def reset(self):
        """Reset filter state."""
        self.x = np.zeros(2)
        self.P = np.eye(2) * 100.0
        self._initialized = False
        self._boost_active = False
        self.Q_current = self.Q_base.copy()

    def update(self, theta_observed: float, confidence: float = 1.0):
        """Kalman update step.

        Args:
            theta_observed: observed source azimuth (deg), i.e.
                theta_local + yaw_current - offset_current
            confidence: 0-1 from localizer (reduces measurement noise)

        Returns:
            (theta_src, omega_src, P_diag) tuple
        """
        z = np.array([theta_observed])

        # Adjust measurement noise by confidence
        R_adj = self.R_base / max(confidence, 0.01)

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q_current

        # Innovation (angular: unwrap to [-180, 180] for azimuth)
        y_raw = z - self.H @ x_pred  # raw residual
        y = np.array([((float(y_raw[0]) + 180) % 360) - 180])
        S = self.H @ P_pred @ self.H.T + R_adj
        K = P_pred @ self.H.T / S  # Kalman gain (scalar for 1D obs)

        # Update
        self.x = x_pred + K.flatten() * y[0]
        self.P = P_pred - np.outer(K, self.H @ P_pred)

        # Adaptive Q: if residual large, boost process noise
        if abs(y[0]) > self.residual_threshold:
            self._boost_active = True
            self.Q_current = self.Q_base * self.Q_boost_factor
        elif self._boost_active:
            self.Q_current *= self.adaption_decay
            if np.max(self.Q_current) < np.max(self.Q_base) * 1.5:
                self.Q_current = self.Q_base.copy()
                self._boost_active = False

        if not self._initialized:
            self._initialized = True

        return float(self.x[0]), float(self.x[1]), np.diag(self.P).copy()

    def predict(self, dt: float):
        """Predict state dt seconds into the future.

        Returns:
            (theta_pred, omega_pred) tuple
        """
        F_pred = np.array([[1, dt],
                           [0, 1]])
        x_pred = F_pred @ self.x
        return float(x_pred[0]), float(x_pred[1])

    @property
    def theta(self) -> float:
        return float(self.x[0])

    @property
    def omega(self) -> float:
        return float(self.x[1])

    @property
    def covariance(self) -> float:
        return float(self.P[0, 0])
