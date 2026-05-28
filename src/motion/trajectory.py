"""Sound source trajectory generation.

Supported types: static, linear, semicircle, circle.
Each generator returns an (N, 3) float64 array of [x, y, z] positions.
"""

import numpy as np

from src.config.schema import MotionConfig


def generate_trajectory(motion: MotionConfig, duration: float,
                        fs: int) -> np.ndarray:
    """Generate trajectory points for the full signal duration.

    Returns:
        np.ndarray of shape (n_samples, 3)
    """
    if not motion.enabled or motion.type == "static":
        return _static(motion.params, duration, fs)

    generators = {
        "linear": _linear,
        "semicircle": _semicircle,
        "circle": _circle,
        "spiral": _spiral,
    }
    gen = generators.get(motion.type, _static)
    return gen(motion.params, duration, fs)


def _static(params: dict, duration: float, fs: int) -> np.ndarray:
    pos = np.array(params.get("position", [2.5, 2.0, 1.5]))
    n_samples = int(duration * fs)
    return np.tile(pos, (n_samples, 1)).astype(np.float64)


def _linear(params: dict, duration: float, fs: int) -> np.ndarray:
    start = np.array(params["start"])
    end = np.array(params["end"])
    speed = params.get("speed", 0.5)

    distance = np.linalg.norm(end - start)
    travel_time = distance / speed if speed > 0 else duration
    n_samples = int(min(travel_time, duration) * fs)

    t = np.linspace(0, 1, n_samples)
    trajectory = start + (end - start) * t[:, np.newaxis]

    # If the source arrives early, hold at the end position
    total_samples = int(duration * fs)
    if n_samples < total_samples:
        static_part = np.tile(end, (total_samples - n_samples, 1))
        trajectory = np.vstack([trajectory, static_part])

    return trajectory.astype(np.float64)


def _semicircle(params: dict, duration: float, fs: int) -> np.ndarray:
    center = np.array(params.get("center", [2.5, 2.0, 1.5]))
    radius = params.get("radius", 1.5)
    speed = params.get("speed", 0.5)
    start_angle = np.deg2rad(params.get("start_angle", 0))
    end_angle = np.deg2rad(params.get("end_angle", 180))

    total_angle = abs(end_angle - start_angle)
    arc_length = radius * total_angle
    travel_time = arc_length / speed if speed > 0 else duration
    actual_duration = min(travel_time, duration)
    n_samples = int(actual_duration * fs)

    angles = np.linspace(start_angle, end_angle, n_samples)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    z = np.full_like(x, center[2])

    return np.column_stack([x, y, z]).astype(np.float64)


def _circle(params: dict, duration: float, fs: int) -> np.ndarray:
    center = np.array(params.get("center", [2.5, 2.0, 1.5]))
    radius = params.get("radius", 1.5)
    speed = params.get("speed", 0.5)

    angular_speed = speed / radius if radius > 0 else 1.0
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    angle = angular_speed * t

    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    z = np.full_like(x, center[2])

    return np.column_stack([x, y, z]).astype(np.float64)


def _spiral(params: dict, duration: float, fs: int) -> np.ndarray:
    center = np.array(params.get("center", [2.5, 2.0, 1.5]))
    start_radius = params.get("start_radius", 0.5)
    end_radius = params.get("end_radius", 2.0)
    revolutions = params.get("revolutions", 2)
    z = params.get("height", center[2])

    n_samples = int(duration * fs)
    t = np.linspace(0, 1, n_samples)
    angle = 2 * np.pi * revolutions * t
    radius = start_radius + (end_radius - start_radius) * t

    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    z_arr = np.full_like(x, z)

    return np.column_stack([x, y, z_arr]).astype(np.float64)
