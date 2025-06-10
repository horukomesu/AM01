"""Camera calibration wrapper built on pycolmap.

This module exposes a ``CameraCalibrator`` class compatible with the previous
API used in ``AMmain.py`` and ``AMUtilities.py``.  It accepts manual feature
tracks and image paths, reconstructs camera poses and 3D points using
``pycolmap`` and exposes several helper methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap


@dataclass
class CameraParameters:
    """Intrinsics, extrinsics and projection matrix for a camera."""

    projection: Optional[np.ndarray]  # 3x4 projection matrix
    rotation: Optional[np.ndarray]    # 3x3 rotation matrix
    translation: Optional[np.ndarray] # 3x1 translation vector
    center: Optional[np.ndarray]      # 3x1 camera center in world coordinates


class PycolmapCalibrator:
    """Minimal SfM pipeline using pycolmap."""

    def __init__(self) -> None:
        self.tracks: List[Dict[int, Tuple[float, float]]] = []
        self.image_paths: List[str] = []
        self.image_shapes: List[Tuple[int, int]] = []

        self.recon: Optional[pycolmap.Reconstruction] = None
        self.track_point_ids: List[Optional[int]] = []
        self.images_info: List[Dict[int, int]] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_tracks(self,
                    tracks: List[Dict[int, Tuple[float, float]]],
                    image_paths: List[str]) -> None:
        """Store feature tracks and image paths."""
        self.tracks = list(tracks)
        self.image_paths = list(image_paths)
        self.image_shapes = []
        for p in self.image_paths:
            with Image.open(p) as img:
                self.image_shapes.append((img.height, img.width))

    # ------------------------------------------------------------------
    # Reconstruction pipeline
    # ------------------------------------------------------------------
    def reconstruct(self) -> None:
        if not self.tracks or not self.image_paths:
            raise ValueError("Tracks and image paths must be provided")

        self._init_reconstruction()
        if self.recon.num_images() < 2:
            raise ValueError("Need at least two images to calibrate")

        self._estimate_initial_pair()
        for idx in range(2, self.recon.num_images() + 1):
            self._register_image(idx)
            self._triangulate_new_points()

        pycolmap.bundle_adjustment(self.recon)
        self.reprojection_error = float(
            self.recon.compute_mean_reprojection_error())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_reconstruction(self) -> None:
        self.recon = pycolmap.Reconstruction()
        self.images_info = []
        self.track_point_ids = [None] * len(self.tracks)

        for idx, path in enumerate(self.image_paths):
            h, w = self.image_shapes[idx]
            f = float(max(h, w))
            cx, cy = w / 2.0, h / 2.0
            cam = pycolmap.Camera(model="SIMPLE_PINHOLE",
                                   width=w,
                                   height=h,
                                   params=[f, cx, cy])
            cam.camera_id = idx + 1
            self.recon.add_camera(cam)

            pts = []
            mapping = {}
            for track_idx, tr in enumerate(self.tracks):
                if idx in tr:
                    mapping[track_idx] = len(pts)
                    pts.append(tr[idx])
            keypoints = np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2), float)
            img = pycolmap.Image(name=str(Path(path).name),
                                 keypoints=keypoints,
                                 camera_id=cam.camera_id)
            img.image_id = idx + 1
            self.recon.add_image(img)
            self.images_info.append(mapping)

    def _estimate_initial_pair(self) -> None:
        img0 = self.recon.image(1)
        img1 = self.recon.image(2)
        cam0 = self.recon.camera(img0.camera_id)
        cam1 = self.recon.camera(img1.camera_id)

        pts0, pts1 = [], []
        for tr in self.tracks:
            if 0 in tr and 1 in tr:
                pts0.append(tr[0])
                pts1.append(tr[1])
        if len(pts0) < 4:
            raise RuntimeError("Not enough correspondences for initial pair")
        pts0 = np.asarray(pts0, dtype=np.float64)
        pts1 = np.asarray(pts1, dtype=np.float64)
        geometry = pycolmap.TwoViewGeometry()
        if not pycolmap.estimate_two_view_geometry_pose(cam0, pts0, cam1, pts1, geometry):
            raise RuntimeError("Failed to estimate initial pose")

        img0.cam_from_world = pycolmap.Rigid3d()
        img1.cam_from_world = geometry.cam2_from_cam1
        self.recon.register_image(img0.image_id)
        self.recon.register_image(img1.image_id)

        for idx, tr in enumerate(self.tracks):
            if 0 in tr and 1 in tr:
                self._triangulate_track(idx)

    def _register_image(self, img_idx: int) -> None:
        if self.recon.is_image_registered(img_idx):
            return
        img = self.recon.image(img_idx)
        cam = self.recon.camera(img.camera_id)

        pts2d, pts3d = [], []
        mapping = self.images_info[img_idx - 1]
        for track_idx, point_idx in mapping.items():
            pid = self.track_point_ids[track_idx]
            if pid is None:
                continue
            pt3d = self.recon.point3D(pid).xyz
            pts3d.append(pt3d)
            pts2d.append(self.tracks[track_idx][img_idx - 1])
        if len(pts3d) < 4:
            return
        pts2d = np.asarray(pts2d, dtype=np.float64)
        pts3d = np.asarray(pts3d, dtype=np.float64)
        est = pycolmap.estimate_absolute_pose(pts2d, pts3d, cam)
        if est is None:
            return
        pose = est["cam_from_world"]
        inliers = est.get("inliers") or est.get("inlier_mask")
        ref = pycolmap.refine_absolute_pose(pose, pts2d, pts3d, inliers, cam)
        if ref is not None:
            pose = ref["cam_from_world"]
        img.cam_from_world = pose
        self.recon.register_image(img_idx)

    def _triangulate_track(self, track_idx: int) -> None:
        if self.track_point_ids[track_idx] is not None:
            return
        tr = self.tracks[track_idx]
        cams = []
        poses = []
        points = []
        obs = []
        for img_idx, xy in tr.items():
            if not self.recon.is_image_registered(img_idx + 1):
                continue
            img = self.recon.image(img_idx + 1)
            cam = self.recon.camera(img.camera_id)
            cams.append(cam)
            poses.append(img.cam_from_world)
            points.append(xy)
            pt_idx = self.images_info[img_idx][track_idx]
            obs.append(pycolmap.TrackElement(image_id=img_idx + 1,
                                             point2D_idx=pt_idx))
        if len(cams) < 2:
            return
        points = np.asarray(points, dtype=np.float64)
        res = pycolmap.estimate_triangulation(points, poses, cams)
        if res is None:
            return
        xyz = res["XYZ"]
        track = pycolmap.Track(obs)
        pid = self.recon.add_point3D(xyz, track)
        for o in obs:
            self.recon.image(o.image_id).set_point3D_for_point2D(o.point2D_idx, pid)
        self.track_point_ids[track_idx] = pid

    def _triangulate_new_points(self) -> None:
        for idx in range(len(self.tracks)):
            if self.track_point_ids[idx] is None:
                self._triangulate_track(idx)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_camera_parameters(self) -> List[CameraParameters]:
        params = []
        for idx in range(1, self.recon.num_images() + 1):
            img = self.recon.image(idx)
            cam = self.recon.camera(img.camera_id)
            if not img.has_pose:
                params.append(CameraParameters(None, None, None, None))
                continue
            R = img.cam_from_world.rotation.matrix()
            t = img.cam_from_world.translation.reshape(3, 1)
            P = cam.calibration_matrix() @ np.hstack((R, t))
            c = -R.T @ t
            params.append(CameraParameters(P, R, t, c))
        return params

    def get_points_3d(self) -> List[np.ndarray]:
        points = []
        for pid in self.recon.point3D_ids():
            points.append(self.recon.point3D(pid).xyz.copy())
        return points

    def get_camera_intrinsics(self) -> List[np.ndarray]:
        mats = []
        for cam_id in self.recon.cameras:
            mats.append(self.recon.cameras[cam_id].calibration_matrix().copy())
        return mats

    def get_reprojection_error(self) -> Optional[float]:
        return getattr(self, "reprojection_error", None)


# Backwards compatibility ------------------------------------------------------
CameraCalibrator = PycolmapCalibrator
