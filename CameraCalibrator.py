"""Automatic camera calibration using pycolmap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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

    def __init__(self, feature_type: str = "SIFT") -> None:
        self.feature_type = feature_type
        self.image_paths: List[str] = []
        self.image_names: List[str] = []
        self.image_shapes: List[tuple[int, int]] = []

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
    def detect_and_match_features(self) -> None:
        """Extract SIFT features and match all image pairs."""
        if not self.image_paths:
            raise ValueError("No images loaded")

        self._temp_dir = tempfile.mkdtemp(prefix="sfm_")
        self._images_dir = os.path.join(self._temp_dir, "images")
        os.makedirs(self._images_dir, exist_ok=True)
        for src, name in zip(self.image_paths, self.image_names):
            shutil.copy(src, os.path.join(self._images_dir, name))

        self._db_path = os.path.join(self._temp_dir, "database.db")
        pycolmap.extract_features(self._db_path, self._images_dir, image_list=self.image_names)
        pycolmap.match_exhaustive(self._db_path)

    # ------------------------------------------------------------------
    def run_reconstruction(self) -> None:
        """Run incremental mapping and bundle adjustment."""
        if self._db_path is None or self._images_dir is None:
            raise RuntimeError("Features were not detected")

        out_path = os.path.join(self._temp_dir, "reconstruction")
        result = pycolmap.incremental_mapping(self._db_path, self._images_dir, out_path)
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

