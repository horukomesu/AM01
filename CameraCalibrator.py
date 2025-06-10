"""Self-calibrating Structure-from-Motion module.

This module implements a very small incremental SfM pipeline that estimates
camera intrinsics, extrinsics and 3D points using only 2D feature tracks.
It is intentionally lightweight and uses only ``numpy``/``opencv``/``scipy`` so
it can be embedded inside 3ds Max or used as a standalone Python module.

Input format
------------
``tracks`` is a list where each element corresponds to a single 3D point.  Each
point is represented by a dictionary mapping ``image_index`` to 2D pixel
coordinates::

    tracks = [
        {0: (x0, y0), 1: (x1, y1)},
        {0: (x2, y2), 2: (x3, y3)},
        ...
    ]

``image_shapes`` is a list of ``(height, width)`` pairs for each image.  No
prior knowledge about the cameras (focal length, sensor width, etc.) is
required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from scipy.optimize import least_squares


@dataclass
class CameraParameters:
    """Intrinsics, extrinsics and projection matrix for a camera."""

    projection: Optional[np.ndarray]  # 3x4 projection matrix
    rotation: Optional[np.ndarray]    # 3x3 rotation matrix
    translation: Optional[np.ndarray] # 3x1 translation vector
    center: Optional[np.ndarray]      # 3x1 camera center in world coordinates


class SelfCalibratingSfM:
    """Incremental SfM solver that tries to recover cameras only from 2D points."""

    def __init__(self) -> None:
        self.tracks: List[Dict[int, Tuple[float, float]]] = []
        self.image_shapes: List[Tuple[int, int]] = []

        self.intrinsics: Optional[np.ndarray] = None
        self.cameras: List[CameraParameters] = []
        self.points_3d: List[Optional[np.ndarray]] = []
        self.reprojection_error: Optional[float] = None

    # ------------------------------------------------------------------
    # Data loading utilities
    # ------------------------------------------------------------------
    def load_tracks(self,
                    tracks: List[Dict[int, Tuple[float, float]]],
                    image_shapes: List[Tuple[int, int]]) -> None:
        """Store feature tracks and image sizes for later reconstruction."""
        self.tracks = list(tracks)
        self.image_shapes = list(image_shapes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reconstruct(self) -> None:
        """Run the full reconstruction pipeline."""
        if not self.tracks or not self.image_shapes:
            raise ValueError("Tracks and image shapes must be provided")

        self.intrinsics = self._initial_intrinsics(self.image_shapes[0])
        self.cameras = []
        self.points_3d = [None] * len(self.tracks)

        # --- Initialize first two cameras ---
        R0 = np.eye(3, dtype=np.float64)
        t0 = np.zeros((3, 1), dtype=np.float64)
        P0 = self.intrinsics @ np.hstack((R0, t0))
        self.cameras.append(CameraParameters(P0, R0, t0, -R0.T @ t0))

        R1, t1 = self._estimate_initial_baseline(0, 1)
        P1 = self.intrinsics @ np.hstack((R1, t1))
        self.cameras.append(CameraParameters(P1, R1, t1, -R1.T @ t1))

        # Triangulate tracks visible in both first cameras
        for idx, tr in enumerate(self.tracks):
            if 0 in tr and 1 in tr:
                X = self._triangulate_point(P0, P1, tr[0], tr[1])
                self.points_3d[idx] = X

        # --- Incrementally add the rest of the cameras ---
        for cam_idx in range(2, len(self.image_shapes)):
            self._add_camera(cam_idx)
            self._triangulate_new_points()

        # --- Refine everything with bundle adjustment ---
        if len(self.get_points_3d()) >= 5 and len(self.cameras) >= 2:
            self._bundle_adjustment()

        self._compute_reprojection_error()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _initial_intrinsics(image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape[:2]
        f = float(max(h, w))
        cx = w / 2.0
        cy = h / 2.0
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    def _estimate_initial_baseline(self, cam_a: int, cam_b: int) -> Tuple[np.ndarray, np.ndarray]:
        pts_a, pts_b = self._collect_correspondences(cam_a, cam_b)
        if len(pts_a) < 8:
            raise ValueError("Not enough correspondences between first two cameras")
        F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC)
        if F is None:
            raise ValueError("Fundamental matrix estimation failed")
        E = self.intrinsics.T @ F @ self.intrinsics
        _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b, self.intrinsics)
        return R, t

    def _add_camera(self, cam_idx: int) -> None:
        pts3d, pts2d = [], []
        for i, tr in enumerate(self.tracks):
            if cam_idx in tr and self.points_3d[i] is not None:
                pts3d.append(self.points_3d[i])
                pts2d.append(tr[cam_idx])
        if len(pts3d) < 4:
            self.cameras.append(CameraParameters(None, None, None, None))
            return
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            np.asarray(pts3d, dtype=np.float64),
            np.asarray(pts2d, dtype=np.float64),
            self.intrinsics,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            self.cameras.append(CameraParameters(None, None, None, None))
            return
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        P = self.intrinsics @ np.hstack((R, t))
        self.cameras.append(CameraParameters(P, R, t, -R.T @ t))

    def _triangulate_new_points(self) -> None:
        for idx, tr in enumerate(self.tracks):
            if self.points_3d[idx] is not None:
                continue
            cams = [i for i, c in enumerate(self.cameras) if c.projection is not None and i in tr]
            if len(cams) < 2:
                continue
            c1, c2 = cams[:2]
            P1, P2 = self.cameras[c1].projection, self.cameras[c2].projection
            X = self._triangulate_point(P1, P2, tr[c1], tr[c2])
            self.points_3d[idx] = X

    def _collect_correspondences(self, cam_a: int, cam_b: int) -> Tuple[np.ndarray, np.ndarray]:
        pts_a, pts_b = [], []
        for tr in self.tracks:
            if cam_a in tr and cam_b in tr:
                pts_a.append(tr[cam_a])
                pts_b.append(tr[cam_b])
        return np.asarray(pts_a, dtype=np.float64), np.asarray(pts_b, dtype=np.float64)

    @staticmethod
    def _triangulate_point(P1: np.ndarray, P2: np.ndarray, pt1, pt2) -> np.ndarray:
        pt1 = np.array(pt1, dtype=np.float64).reshape(2, 1)
        pt2 = np.array(pt2, dtype=np.float64).reshape(2, 1)
        X_h = cv2.triangulatePoints(P1, P2, pt1, pt2)
        X = (X_h[:3] / X_h[3]).reshape(3)
        return X

    # ------------------------------------------------------------------
    # Bundle adjustment
    # ------------------------------------------------------------------
    def _bundle_adjustment(self) -> None:
        cam_indices = {i: idx for idx, i in enumerate(range(1, len(self.cameras)))}
        point_indices = {j: idx for idx, j in enumerate(i for i, p in enumerate(self.points_3d) if p is not None)}

        def pack_parameters() -> np.ndarray:
            f = float(self.intrinsics[0, 0])
            cx = float(self.intrinsics[0, 2])
            cy = float(self.intrinsics[1, 2])
            params = [f, cx, cy]
            for i in range(1, len(self.cameras)):
                cam = self.cameras[i]
                if cam.projection is None:
                    params.extend([0.0] * 6)
                    continue
                rvec, _ = cv2.Rodrigues(cam.rotation)
                params.extend(rvec.ravel())
                params.extend(cam.translation.ravel())
            for p in self.points_3d:
                if p is not None:
                    params.extend(p.ravel())
            return np.asarray(params, dtype=np.float64)

        def unpack_parameters(x: np.ndarray):
            f, cx, cy = x[:3]
            K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            offset = 3
            Rs, ts = [], []
            for i in range(1, len(self.cameras)):
                rvec = x[offset:offset + 3]
                t = x[offset + 3:offset + 6].reshape(3, 1)
                offset += 6
                R, _ = cv2.Rodrigues(rvec)
                Rs.append(R)
                ts.append(t)
            points = []
            for _ in range(len(point_indices)):
                pt = x[offset:offset + 3]
                points.append(pt)
                offset += 3
            return K, Rs, ts, points

        def residuals(x: np.ndarray) -> np.ndarray:
            K, Rs, ts, pts = unpack_parameters(x)
            # Build projection matrices
            Ps = [K @ np.hstack((np.eye(3), np.zeros((3, 1))))]
            for R, t in zip(Rs, ts):
                Ps.append(K @ np.hstack((R, t)))
            # Map points
            pts_full = {}
            for j, idx in point_indices.items():
                pts_full[j] = pts[idx]
            # Residual list
            res = []
            for j, tr in enumerate(self.tracks):
                if j not in pts_full:
                    continue
                X = np.hstack((pts_full[j], 1.0))
                for cam_idx, uv in tr.items():
                    if cam_idx >= len(Ps) or Ps[cam_idx] is None:
                        continue
                    proj = Ps[cam_idx] @ X
                    proj = proj[:2] / proj[2]
                    res.extend(proj - np.asarray(uv))
            return np.asarray(res, dtype=np.float64)

        x0 = pack_parameters()
        result = least_squares(residuals, x0, method="trf", verbose=0, max_nfev=200)
        K, Rs, ts, pts = unpack_parameters(result.x)
        self.intrinsics = K
        # update cameras
        for i in range(1, len(self.cameras)):
            cam = self.cameras[i]
            if cam.projection is None:
                continue
            R = Rs[i - 1]
            t = ts[i - 1]
            P = K @ np.hstack((R, t))
            self.cameras[i] = CameraParameters(P, R, t, -R.T @ t)
        # update points
        for j, idx in point_indices.items():
            self.points_3d[j] = pts[idx]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _compute_reprojection_error(self) -> None:
        total = 0.0
        count = 0
        K = self.intrinsics
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        for i, tr in enumerate(self.tracks):
            X = self.points_3d[i]
            if X is None:
                continue
            X_h = np.hstack((X, 1.0))
            for cam_idx, uv in tr.items():
                if cam_idx == 0:
                    P = P0
                elif cam_idx < len(self.cameras):
                    cam = self.cameras[cam_idx]
                    if cam.projection is None:
                        continue
                    P = cam.projection
                else:
                    continue
                proj = P @ X_h
                proj = proj[:2] / proj[2]
                total += float(np.linalg.norm(proj - np.asarray(uv)))
                count += 1
        self.reprojection_error = total / count if count else None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_camera_parameters(self) -> List[CameraParameters]:
        return list(self.cameras)

    def get_points_3d(self) -> List[np.ndarray]:
        return [p for p in self.points_3d if p is not None]

    def get_camera_intrinsics(self) -> Optional[np.ndarray]:
        return None if self.intrinsics is None else self.intrinsics.copy()

    def get_reprojection_error(self) -> Optional[float]:
        return self.reprojection_error

    # ------------------------------------------------------------------
    # Optional visualization
    # ------------------------------------------------------------------
    def plot_reconstruction(self) -> None:
        """Display simple 3D scatter plot of cameras and points."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # unused
        except Exception:  # pragma: no cover - matplotlib optional
            return

        cam_centers = [c.center.reshape(3) for c in self.cameras if c.center is not None]
        pts = [p for p in self.points_3d if p is not None]
        if not cam_centers or not pts:
            return
        cam_centers = np.asarray(cam_centers)
        pts = np.asarray(pts)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="r", label="points")
        ax.scatter(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2], c="b", label="cameras")
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()


# Backwards compatibility ------------------------------------------------------
# Retain the old class name so existing code importing ``CameraCalibrator``
# continues to work.
CameraCalibrator = SelfCalibratingSfM
