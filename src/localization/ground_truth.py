"""Compute ground-truth DOA from known trajectory and head position."""

import numpy as np


def compute_truth_doa(trajectory, head_center):
    """Compute true azimuth angle for each trajectory point relative to head.

    Azimuth: 0 = straight ahead (positive y), +90 = right, -90 = left.
    Uses the interaural axis (x-axis) as reference.

    Args:
        trajectory: (N, 3) source positions [x, y, z]
        head_center: (3,) head center [x, y, z]

    Returns:
        (N,) array of azimuth angles in degrees
    """
    traj = np.atleast_2d(trajectory)
    head = np.atleast_1d(head_center)

    rel = traj - head  # (N, 3) relative position

    # Azimuth: angle in horizontal plane relative to straight-ahead (y-axis)
    # atan2(x, y): 0 = ahead, +90 = right, -90 = left
    azimuth_rad = np.arctan2(rel[:, 0], rel[:, 1])

    return np.rad2deg(azimuth_rad)


def compute_truth_elevation(trajectory, head_center):
    """Compute true elevation angle for each trajectory point.

    Elevation: 0 = horizontal, +90 = zenith, -90 = nadir.

    Args:
        trajectory: (N, 3) source positions
        head_center: (3,) head center

    Returns:
        (N,) array of elevation angles in degrees
    """
    traj = np.atleast_2d(trajectory)
    head = np.atleast_1d(head_center)

    rel = traj - head
    dist_h = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
    elev_rad = np.arctan2(rel[:, 2], dist_h)

    return np.rad2deg(elev_rad)


def compute_true_azimuth_woodworth(azimuth_deg):
    """Convert Cartesian azimuth to Woodworth-model azimuth.

    Woodworth ITD model uses angle from interaural axis (0 = ear-level lateral).

    Args:
        azimuth_deg: geometric azimuth in degrees

    Returns:
        Woodworth azimuth in radians [0, pi/2]
    """
    az = np.deg2rad(np.clip(np.abs(azimuth_deg), 0, 180))
    return np.where(az > np.pi/2, np.pi - az, az)
