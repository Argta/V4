"""Metric functions for localization error evaluation."""

import numpy as np


def angular_error(estimated_deg, truth_deg):
    """Angular error wrapping to [-180, 180].

    Args:
        estimated_deg: scalar or array of estimated azimuth in degrees
        truth_deg: scalar or array of true azimuth in degrees

    Returns:
        angular difference in degrees, wrapped to [-180, 180]
    """
    diff = np.atleast_1d(np.array(estimated_deg) - np.array(truth_deg))
    return ((diff + 180) % 360) - 180


def rmse(errors_deg):
    """Root mean square angular error.

    Args:
        errors_deg: array of angular errors in degrees

    Returns:
        RMSE in degrees
    """
    return float(np.sqrt(np.mean(np.square(errors_deg))))


def mae(errors_deg):
    """Mean absolute angular error."""
    return float(np.mean(np.abs(errors_deg)))


def accuracy_within(errors_deg, threshold_deg=10.0):
    """Fraction of frames with error within threshold.

    Args:
        errors_deg: array of angular errors
        threshold_deg: error tolerance in degrees

    Returns:
        float in [0, 1]
    """
    return float(np.mean(np.abs(errors_deg) <= threshold_deg))


def front_back_confusion_rate(estimated_deg, truth_deg):
    """Compute front-back confusion rate.

    Front-back confusion: estimated angle is in the wrong hemifield
    (front vs back). For azimuth: front = [-90, 90], back = [90, 180] or [-180, -90].

    Returns:
        float in [0, 1]: fraction of frames with front-back confusion
    """
    est = np.atleast_1d(np.array(estimated_deg))
    truth = np.atleast_1d(np.array(truth_deg))

    def is_front(az):
        return np.abs(az) <= 90

    est_front = is_front(est)
    truth_front = is_front(truth)

    confused = est_front != truth_front
    return float(np.mean(confused))


def confusion_matrix_quadrants(estimated_deg, truth_deg):
    """Build a 4x4 confusion matrix for DOA quadrants.

    Quadrants:
        0: Front-Right   (0 to 90)
        1: Front-Left    (-90 to 0)
        2: Back-Right    (90 to 180)
        3: Back-Left     (-180 to -90)

    Returns:
        4x4 numpy array, rows=truth, cols=estimated
    """
    est = np.atleast_1d(np.array(estimated_deg))
    truth = np.atleast_1d(np.array(truth_deg))

    def quadrant(az):
        az = np.atleast_1d(az)
        q = np.zeros(len(az), dtype=int)
        # Use vectorized conditions
        q[(az >= 0) & (az <= 90)] = 0    # Front-Right
        q[(az < 0) & (az >= -90)] = 1    # Front-Left
        q[(az > 90) & (az <= 180)] = 2   # Back-Right
        q[(az < -90) & (az >= -180)] = 3  # Back-Left
        return q

    q_est = quadrant(est)
    q_truth = quadrant(truth)

    cm = np.zeros((4, 4), dtype=int)
    for t in range(4):
        mask = q_truth == t
        for e in range(4):
            cm[t, e] = int(np.sum(mask & (q_est == e)))

    return cm


def compute_all_metrics(estimated_deg, truth_deg):
    """Compute all standard localization metrics.

    Args:
        estimated_deg: (N,) estimated azimuth in degrees
        truth_deg: (N,) true azimuth in degrees

    Returns:
        dict with keys: rmse, mae, accuracy_5deg, accuracy_10deg,
                        accuracy_30deg, front_back_confusion,
                        confusion_matrix, n_frames
    """
    errors = angular_error(estimated_deg, truth_deg)
    return {
        "rmse": rmse(errors),
        "mae": mae(errors),
        "accuracy_5deg": accuracy_within(errors, 5.0),
        "accuracy_15deg": accuracy_within(errors, 15.0),
        "accuracy_30deg": accuracy_within(errors, 30.0),
        "front_back_confusion": front_back_confusion_rate(estimated_deg, truth_deg),
        "confusion_matrix": confusion_matrix_quadrants(estimated_deg, truth_deg).tolist(),
        "n_frames": len(errors),
    }
