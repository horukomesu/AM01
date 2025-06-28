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

import numpy as np


from CameraCalibrator import CameraCalibrator
try:
    from PySide2 import QtGui
except ImportError:
    from PySide6 import QtGui

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
) -> None:
    import pymxs
    import math
    import numpy as np
    from pathlib import Path

    rt = pymxs.runtime

    OPENCV_TO_MAX = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ], dtype=np.float32)

    CAMERA_ROT = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)

    def as_point3(vec: np.ndarray):
        return rt.Point3(float(vec[0]), float(vec[1]), float(vec[2]))

    if not calibrator.calibration_results:
        raise RuntimeError("No calibration results found.")

    results = calibrator.calibration_results
    poses = results.get("poses", {})
    intrinsics = results.get("intrinsics", {})
    points_3d = results.get("points_3d", [])
    registered_indices = results.get("registered_indices", [])

    # --- Экспорт локаторов (3D точек) ---
    if len(locator_names) < len(points_3d):
        locator_names += [f"pt{i}" for i in range(len(locator_names), len(points_3d))]

    for pt, name in zip(points_3d, locator_names):
        pt_cv = np.array(pt, dtype=np.float32).reshape(3)
        pt_max = OPENCV_TO_MAX @ pt_cv
        dummy = rt.Dummy(name=name)
        dummy.position = as_point3(pt_max)

    # --- Экспорт камер ---
    for img_idx in registered_indices:
        if img_idx not in poses or img_idx not in intrinsics:
            continue

        pose = poses[img_idx]
        R = np.array(pose["R"], dtype=np.float32)
        t = np.array(pose["t"], dtype=np.float32).reshape(3)

        R_max = OPENCV_TO_MAX @ R @ CAMERA_ROT
        t_max = OPENCV_TO_MAX @ t

        # Сборка трансформа
        T = np.eye(4)
        T[:3, :3] = R_max
        T[:3, 3] = t_max

        tm = rt.matrix3(
            as_point3(T[:3, 0]),
            as_point3(T[:3, 1]),
            as_point3(T[:3, 2]),
            as_point3(T[:3, 3]),
        )

        # Создание камеры
        cam_name = f"Cam_{Path(image_paths[img_idx]).stem}"
        cam = rt.FreeCamera(name=cam_name)

        # Вычисление FOV (3ds Max требует горизонтальный угол)
        K = np.array(intrinsics[img_idx]["K"], dtype=np.float32)
        f = K[0, 0]  # fx
        width = calibrator.image_shapes[img_idx][0]

        xfov_rad = 2 * math.atan((width / 2.0) / f)
        xfov_deg = math.degrees(xfov_rad)

        cam.fov = xfov_deg
        cam.transform = tm
