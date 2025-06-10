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

FORMAT_VERSION = 1

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

def save_scene(
    image_paths: List[str],
    locators: List[Dict[str, Any]],
    path: str
) -> None:
    """Сохраняет текущую сцену в JSON-файл.

    Parameters
    ----------
    image_paths : List[str]
        Список путей к изображениям.
    locators : List[Dict[str, Any]]
        Список локаторов (имя, позиции на изображениях).
    path : str
        Путь к файлу для сохранения.
    """
    scene = {
        "format_version": FORMAT_VERSION,
        "images": list(image_paths),
        "locators": list(locators),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh, indent=2, ensure_ascii=False)

def load_scene(path: str) -> Dict[str, Any]:
    """Загружает сцену из JSON. Проверяет структуру и версию.

    Parameters
    ----------
    path : str
        Путь к сцене.

    Returns
    -------
    Dict[str, Any]
        Словарь с ключами: "images", "locators".
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Scene file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        scene = json.load(fh)

    # Проверка структуры
    if not isinstance(scene, dict):
        raise ValueError("Scene file format error: not a dict")
    if "images" not in scene or "locators" not in scene:
        raise ValueError("Scene file missing required fields: images or locators")
    if scene.get("format_version", 1) > FORMAT_VERSION:
        raise ValueError("Scene file format version is newer than supported!")

    # Проверка путей
    image_paths = [str(p) for p in scene.get("images", []) if Path(p).is_file()]
    locators = list(scene.get("locators", []))

    # --- ВАЖНО! --- Исправляем типы ключей у positions (преобразуем в int)
    for loc in locators:
        if "name" not in loc or "positions" not in loc:
            raise ValueError(f"Bad locator entry: {loc}")
        # Привести ключи positions к int, значения оставить как есть
        if isinstance(loc["positions"], dict):
            loc["positions"] = {
                int(k): v for k, v in loc["positions"].items()
            }

    return {
        "images": image_paths,
        "locators": locators,
    }


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

    # Перевод системы координат из OpenCV (Y-down) в 3ds Max (Z-up)
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

        # Явно конвертируем к float!
        def as_float_tuple(arr):
            return float(arr[0]), float(arr[1]), float(arr[2])

        tm = rt.Matrix3(
            rt.Point3(*as_float_tuple(rot[:, 0])),
            rt.Point3(*as_float_tuple(rot[:, 1])),
            rt.Point3(*as_float_tuple(rot[:, 2])),
            rt.Point3(*as_float_tuple(pos)),
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
        dummy.position = rt.Point3(*as_float_tuple(pos))

