"""Camera calibration routines for MyImageModelerPlugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2


@dataclass
class CalibrationResult:
    """Container for calibration results."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    reprojection_error: float


def calibrate_cameras(image_points: List[np.ndarray],
                      object_points: List[np.ndarray],
                      image_size: Tuple[int, int]) -> CalibrationResult:
    """Calibrate cameras using corresponding image and object points."""

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    # Compute reprojection error
    error = 0.0
    total_points = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        e = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2)
        n = len(imgpoints2)
        error += e ** 2
        total_points += n
    reproj_error = (error / total_points) ** 0.5 if total_points else 0.0

    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        reprojection_error=reproj_error,
    )
