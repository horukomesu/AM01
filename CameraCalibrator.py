"""Simple structure-from-motion calibration module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2


@dataclass
class CameraParameters:
    """Projection and extrinsic parameters for a camera."""

    projection: np.ndarray  # 3x4 projection matrix
    rotation: np.ndarray    # 3x3 rotation matrix
    translation: np.ndarray # 3x1 translation vector
    center: np.ndarray      # 3x1 camera center in world coordinates


class CameraCalibrator:
    """Incremental SfM calibrator using only 2D points."""

    def __init__(self, camera_matrix: Optional[np.ndarray] = None) -> None:
        self._camera_matrix = camera_matrix
        self._cameras: List[CameraParameters] = []
        self._points_3d: List[Optional[np.ndarray]] = []
        self._reprojection_error: Optional[float] = None

    @staticmethod
    def _default_camera_matrix(image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape[:2]
        f = max(h, w)
        cx = w / 2.0
        cy = h / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    def calibrate(self, matches: List[Dict[int, Tuple[float, float]]],
                  image_shapes: List[Tuple[int, int]]) -> None:
        """Estimate camera poses and 3D points.

        Parameters
        ----------
        matches : List[Dict[int, Tuple[float, float]]]
            For each 3D point a dictionary mapping camera index to 2D position
            (x, y) in pixel coordinates. Cameras are referenced by their index
            in ``image_shapes``. Points may be missing in some cameras.
        image_shapes : List[Tuple[int, int]]
            Image sizes as ``(height, width)`` for each camera.
        """

        num_cams = len(image_shapes)
        if self._camera_matrix is None:
            self._camera_matrix = self._default_camera_matrix(image_shapes[0])

        K = self._camera_matrix

        # Camera 0 at origin
        R0 = np.eye(3)
        t0 = np.zeros((3, 1))
        P0 = K @ np.hstack((R0, t0))
        self._cameras = [CameraParameters(P0, R0, t0, -R0.T @ t0)]

        # Find initial camera 1 using correspondences with camera 0
        pts0, pts1 = self._collect_correspondences(matches, 0, 1)
        if len(pts0) < 5:
            raise ValueError("Not enough correspondences between first two cameras")

        E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R1, t1, mask_pose = cv2.recoverPose(E, pts0, pts1, K)

        P1 = K @ np.hstack((R1, t1))
        self._cameras.append(CameraParameters(P1, R1, t1, -R1.T @ t1))

        # Triangulate points seen by cameras 0 and 1
        self._points_3d = [None] * len(matches)
        for idx, m in enumerate(matches):
            if 0 in m and 1 in m:
                X = self._triangulate_point(P0, P1, m[0], m[1])
                self._points_3d[idx] = X

        # Process remaining cameras
        for cam_idx in range(2, num_cams):
            pts3d, pts2d, indices = self._collect_pnp_correspondences(matches, cam_idx)
            if len(pts3d) < 4:
                # insufficient data to solve PnP
                self._cameras.append(CameraParameters(None, None, None, None))
                continue
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.asarray(pts3d, dtype=np.float64),
                np.asarray(pts2d, dtype=np.float64),
                K,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                self._cameras.append(CameraParameters(None, None, None, None))
                continue
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
            P = K @ np.hstack((R, t))
            self._cameras.append(CameraParameters(P, R, t, -R.T @ t))
            # Triangulate new points using camera 0 and current camera
            for idx, m in enumerate(matches):
                if self._points_3d[idx] is not None:
                    continue
                if 0 in m and cam_idx in m:
                    X = self._triangulate_point(P0, P, m[0], m[cam_idx])
                    self._points_3d[idx] = X

        self._compute_reprojection_error(matches)

    def _collect_correspondences(self, matches, cam_a, cam_b):
        pts_a = []
        pts_b = []
        for m in matches:
            if cam_a in m and cam_b in m:
                pts_a.append(m[cam_a])
                pts_b.append(m[cam_b])
        return np.array(pts_a, dtype=np.float64), np.array(pts_b, dtype=np.float64)

    def _collect_pnp_correspondences(self, matches, cam_idx):
        pts3d = []
        pts2d = []
        indices = []
        for i, m in enumerate(matches):
            if cam_idx in m and self._points_3d[i] is not None:
                pts3d.append(self._points_3d[i])
                pts2d.append(m[cam_idx])
                indices.append(i)
        return pts3d, pts2d, indices

    @staticmethod
    def _triangulate_point(P1, P2, pt1, pt2):
        pt1 = np.array(pt1, dtype=np.float64).reshape(2, 1)
        pt2 = np.array(pt2, dtype=np.float64).reshape(2, 1)
        X_hom = cv2.triangulatePoints(P1, P2, pt1, pt2)
        X = (X_hom[:3] / X_hom[3]).reshape(3)
        return X

    def _compute_reprojection_error(self, matches):
        total_err = 0.0
        count = 0
        for cam_idx, cam in enumerate(self._cameras):
            if cam.projection is None:
                continue
            P = cam.projection
            for idx, m in enumerate(matches):
                if cam_idx in m and self._points_3d[idx] is not None:
                    X = np.hstack((self._points_3d[idx], 1))
                    proj = P @ X.reshape(4, 1)
                    proj = proj[:2] / proj[2]
                    err = np.linalg.norm(proj.ravel() - np.array(m[cam_idx]))
                    total_err += err
                    count += 1
        self._reprojection_error = total_err / count if count else None

    def get_camera_parameters(self) -> List[CameraParameters]:
        return self._cameras

    def get_points_3d(self) -> List[np.ndarray]:
        return [p for p in self._points_3d if p is not None]

    def get_reprojection_error(self) -> Optional[float]:
        return self._reprojection_error
