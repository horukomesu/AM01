"""Automatic camera calibration using pycolmap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap


@dataclass
class CameraParameters:
    """Intrinsics, extrinsics and projection matrix for a camera."""

    projection: Optional[np.ndarray]
    rotation: Optional[np.ndarray]
    translation: Optional[np.ndarray]
    center: Optional[np.ndarray]


class CameraCalibrator:
    """Simple SfM pipeline based on pycolmap."""

    def __init__(self) -> None:
        """Initialize the calibrator."""
        self.image_paths: List[str] = []
        self.image_names: List[str] = []
        self.image_shapes: List[tuple[int, int]] = []
        self.locators: List[Dict[str, Dict[int, Dict[str, float]]]] = []

        self._temp_dir: Optional[str] = None
        self._db_path: Optional[str] = None
        self._images_dir: Optional[str] = None
        self._recon: Optional[pycolmap.Reconstruction] = None
        self._reproj_error: Optional[float] = None

    # ------------------------------------------------------------------
    def load_images(self, image_paths: List[str]) -> None:
        """Store image paths and read their sizes."""
        self.image_paths = list(image_paths)
        self.image_names = [Path(p).name for p in self.image_paths]
        self.image_shapes = []
        for p in self.image_paths:
            with Image.open(p) as img:
                self.image_shapes.append((img.height, img.width))

    # ------------------------------------------------------------------
    def load_locators(self, locators: List[dict]) -> None:
        """Store user defined locator tracks."""
        self.locators = [dict(l) for l in locators]

    # ------------------------------------------------------------------
    def detect_and_match_features(self) -> None:
        """Create COLMAP database from manually placed points."""
        if not self.image_paths:
            raise ValueError("No images loaded")
        if not self.locators:
            raise ValueError("No locator data provided")

        self._temp_dir = tempfile.mkdtemp(prefix="sfm_")
        self._images_dir = os.path.join(self._temp_dir, "images")
        os.makedirs(self._images_dir, exist_ok=True)
        for src, name in zip(self.image_paths, self.image_names):
            shutil.copy(src, os.path.join(self._images_dir, name))

        self._db_path = os.path.join(self._temp_dir, "database.db")
        db = pycolmap.Database.connect(self._db_path)

        image_id_map = {}
        kp_maps: List[Dict[int, int]] = []

        for idx, (name, shape) in enumerate(zip(self.image_names, self.image_shapes)):
            h, w = shape
            cam = pycolmap.Camera.create("SIMPLE_PINHOLE", w, h, np.array([w, w / 2.0, h / 2.0]))
            cam_id = db.write_camera(cam)
            img = pycolmap.Image(camera_id=cam_id, name=name)
            img_id = db.write_image(img)
            image_id_map[idx] = img_id

            keypoints = []
            kp_map = {}
            for loc_idx, loc in enumerate(self.locators):
                pos = loc.get("positions", {}).get(idx)
                if not pos:
                    continue
                keypoints.append([pos["x"] * w, pos["y"] * h, 1.0, 0.0])
                kp_map[loc_idx] = len(keypoints) - 1

            kp_maps.append(kp_map)
            kp_arr = np.asarray(keypoints, dtype=np.float32)
            if kp_arr.size:
                db.write_keypoints(img_id, kp_arr)
                db.write_descriptors(img_id, np.zeros((len(kp_arr), 128), dtype=np.uint8))

        num_images = len(self.image_paths)
        for i in range(num_images):
            for j in range(i + 1, num_images):
                matches = []
                for loc_idx in range(len(self.locators)):
                    i_idx = kp_maps[i].get(loc_idx)
                    j_idx = kp_maps[j].get(loc_idx)
                    if i_idx is not None and j_idx is not None:
                        matches.append([i_idx, j_idx])
                if matches:
                    arr = np.asarray(matches, dtype=np.uint32)
                    db.write_matches(image_id_map[i], image_id_map[j], arr)
                    tv = pycolmap.TwoViewGeometry()
                    tv.inlier_matches = arr
                    db.write_two_view_geometry(image_id_map[i], image_id_map[j], tv)

        db.close()

    # ------------------------------------------------------------------
    def run_reconstruction(self) -> None:
        """Run incremental mapping and bundle adjustment."""
        if self._db_path is None or self._images_dir is None:
            raise RuntimeError("Features were not detected")

        out_path = os.path.join(self._temp_dir, "reconstruction")
        opts = {
            "Mapper.tri_min_angle": 1.0,
            "Mapper.abs_pose_min_inlier_ratio": 0.1,
        }
        result = pycolmap.incremental_mapping(self._db_path, self._images_dir, out_path, opts)
        if not result:
            raise RuntimeError("Reconstruction failed")
        self._recon = next(iter(result.values()))
        self._reproj_error = float(self._recon.compute_mean_reprojection_error())

    # ------------------------------------------------------------------
    def get_camera_parameters(self) -> List[CameraParameters]:
        params: List[CameraParameters] = []
        if self._recon is None:
            return params
        for name in self.image_names:
            img = self._recon.find_image_with_name(name)
            if img is None or not img.has_pose:
                params.append(CameraParameters(None, None, None, None))
                continue
            cam = self._recon.camera(img.camera_id)
            R = img.cam_from_world.rotation.matrix()
            t = img.cam_from_world.translation.reshape(3, 1)
            P = cam.calibration_matrix() @ np.hstack((R, t))
            c = -R.T @ t
            params.append(CameraParameters(P, R, t, c))
        return params

    # ------------------------------------------------------------------
    def get_points_3d(self) -> List[np.ndarray]:
        if self._recon is None:
            return []
        return [self._recon.point3D(pid).xyz.copy() for pid in self._recon.point3D_ids()]

    # ------------------------------------------------------------------
    def get_camera_intrinsics(self) -> List[np.ndarray]:
        mats = []
        if self._recon is None:
            return mats
        for name in self.image_names:
            img = self._recon.find_image_with_name(name)
            if img is None:
                mats.append(None)
            else:
                cam = self._recon.camera(img.camera_id)
                mats.append(cam.calibration_matrix().copy())
        return mats

    # ------------------------------------------------------------------
    def get_reprojection_error(self) -> Optional[float]:
        return self._reproj_error

    # ------------------------------------------------------------------
    @property
    def recon(self) -> Optional[pycolmap.Reconstruction]:
        return self._recon

