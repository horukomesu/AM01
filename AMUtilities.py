"""Utility functions for MyImageModelerPlugin.

This module holds helper routines used across the application
such as loading images and reading/writing scene description files.

The functions are kept independent from any UI logic so they can be
tested in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import math
import numpy as np

from CameraCalibrator import CameraCalibrator

from PySide2 import QtGui


def load_images(paths: List[str]) -> List[QtGui.QImage]:
    """Load images from disk.

    Parameters
    ----------
    paths : List[str]
        Paths to image files.

    Returns
    -------
    List[QtGui.QImage]
        Loaded images. Missing images are ignored.
    """
    images = []
    for p in paths:
        qimg = QtGui.QImage(p)
        if not qimg.isNull():
            images.append(qimg)
    return images


def save_scene(scene: Dict[str, Any], path: str) -> None:
    """Save the scene dictionary as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh, indent=2)


def load_scene(path: str) -> Dict[str, Any]:
    """Load a scene description from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_paths(paths: List[str]) -> List[str]:
    """Return only existing file paths."""
    return [p for p in paths if Path(p).exists()]


def export_calibration_to_max(
    calibrator: CameraCalibrator,
    image_paths: List[str],
    locator_names: List[str],
    sensor_width_mm: float = 36.0,
) -> None:
    """Create cameras and dummies in the active 3ds Max scene.

    Parameters
    ----------
    calibrator : CameraCalibrator
        Calibrator containing recovered cameras and 3D points.
    image_paths : List[str]
        List of image paths corresponding to cameras. Used for naming.
    locator_names : List[str]
        Names of locators that correspond to the ``matches`` passed to
        :meth:`CameraCalibrator.calibrate`.
    sensor_width_mm : float, optional
        Sensor width in millimeters, by default ``36.0``.
    """

    import pymxs

    rt = pymxs.runtime

    axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

    cams = calibrator.get_camera_parameters()
    K = calibrator.get_camera_intrinsics()
    shapes = calibrator.get_image_shapes()

    for idx, cam in enumerate(cams):
        if cam.center is None or cam.rotation is None:
            continue
        name = Path(image_paths[idx]).stem if idx < len(image_paths) else f"cam{idx}"

        pos = axes @ cam.center.reshape(3)
        rot = axes @ cam.rotation.T

        tm = rt.Matrix3(
            rt.Point3(*rot[:, 0]),
            rt.Point3(*rot[:, 1]),
            rt.Point3(*rot[:, 2]),
            rt.Point3(*pos),
        )
        node = rt.FreeCamera()
        node.name = name
        node.transform = tm

        if K is not None and idx < len(shapes):
            width_px = shapes[idx][1]
            fx = float(K[0, 0])
            focal_mm = fx / width_px * sensor_width_mm
            try:
                node.fov = 2 * math.atan(sensor_width_mm / (2 * focal_mm))
            except AttributeError:
                pass

    for name, pt in calibrator.get_named_points_3d(locator_names):
        pos = axes @ pt.reshape(3)
        dummy = rt.Dummy()
        dummy.name = name
        dummy.position = rt.Point3(*pos)
