"""Tests for trajectory generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config.schema import MotionConfig
from src.motion.trajectory import generate_trajectory


def test_static():
    cfg = MotionConfig(enabled=False, type="static",
                       params={"position": [3, 4, 1.5]})
    traj = generate_trajectory(cfg, duration=0.5, fs=100)
    assert traj.shape == (50, 3)
    np.testing.assert_allclose(traj[0], [3, 4, 1.5])


def test_linear():
    cfg = MotionConfig(enabled=True, type="linear",
                       params={"start": [0, 0, 0], "end": [5, 0, 0], "speed": 2.0})
    traj = generate_trajectory(cfg, duration=1.0, fs=100)
    assert traj.shape[0] > 0
    assert traj.shape[1] == 3
    # Should reach end position
    np.testing.assert_allclose(traj[-1], [5, 0, 0], atol=0.1)


def test_semicircle():
    cfg = MotionConfig(enabled=True, type="semicircle",
                       params={"center": [2.5, 2, 1.5], "radius": 1.5,
                               "speed": 0.5, "start_angle": 0, "end_angle": 180})
    traj = generate_trajectory(cfg, duration=1.0, fs=100)
    assert traj.shape[1] == 3
    # All points should be at z = center_z
    np.testing.assert_allclose(traj[:, 2], 1.5)


def test_circle():
    cfg = MotionConfig(enabled=True, type="circle",
                       params={"center": [4, 3, 1.5], "radius": 1.2, "speed": 0.5})
    traj = generate_trajectory(cfg, duration=2.0, fs=50)
    assert traj.shape[1] == 3
    # Check constant distance from center
    distances = np.linalg.norm(traj[:, :2] - [4, 3], axis=1)
    np.testing.assert_allclose(distances, 1.2, rtol=0.01)


if __name__ == "__main__":
    test_static()
    test_linear()
    test_semicircle()
    test_circle()
    print("All trajectory tests passed!")
