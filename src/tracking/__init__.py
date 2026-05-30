"""Source tracking: Kalman filter + offset controller for continuous tracking.

Eliminates the median-plane dead-zone and align-then-stop behavior
by maintaining a deliberate offset angle between head and source.
"""

from .kalman_tracker import KalmanBinauralTracker
from .offset_controller import OffsetController, TrackingState
from .head_controller import HeadController, ControlMode

__all__ = [
    "KalmanBinauralTracker",
    "OffsetController",
    "TrackingState",
    "HeadController",
    "ControlMode",
]
