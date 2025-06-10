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
    calibrator,
    image_paths,
    locator_names,
    sensor_width_mm: float = 36.0,
):
    """
    Экспортирует камеры и локаторы в текущую сцену 3ds Max (через pymxs).

    calibrator: CameraCalibrator (твой новый класс, SelfCalibratingSfM)
    image_paths: список путей к изображениям (для имён камер)
    locator_names: имена локаторов (для dummy)
    sensor_width_mm: ширина сенсора для пересчёта фокусного (по умолчанию 36 мм)
    """
    import pymxs
    import numpy as np
    from pathlib import Path
    import math

    rt = pymxs.runtime

    # OpenCV → 3ds Max (Y-down → Z-up)
    axes = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ])

    cams = calibrator.get_camera_parameters()
    K = calibrator.get_camera_intrinsics()
    shapes = getattr(calibrator, "image_shapes", None)

    def as_float_tuple(arr):
        return float(arr[0]), float(arr[1]), float(arr[2])

    for idx, cam in enumerate(cams):
        if cam.center is None or cam.rotation is None:
            continue
        name = Path(image_paths[idx]).stem if idx < len(image_paths) else f"cam{idx}"

        # Переводим координаты и ориентацию камеры из OpenCV в Max
        pos = axes @ cam.center.reshape(3)
        rot = axes @ cam.rotation.T

        tm = rt.Matrix3(
            rt.Point3(*as_float_tuple(rot[:, 0])),
            rt.Point3(*as_float_tuple(rot[:, 1])),
            rt.Point3(*as_float_tuple(rot[:, 2])),
            rt.Point3(*as_float_tuple(pos)),
        )
        node = rt.FreeCamera()
        node.name = name
        node.transform = tm

        # Пересчёт focal length (focal_px → focal_mm) → FOV
        if K is not None and idx < len(cams):
            width_px = shapes[idx][1] if shapes is not None else None
            fx = float(K[0, 0])
            if width_px:
                # focal_mm = fx / width_px * sensor_width_mm
                try:
                    # В 3ds Max fov задаётся в радианах!
                    focal_mm = fx / width_px * sensor_width_mm
                    node.fov = 2 * math.atan(sensor_width_mm / (2 * focal_mm))
                except Exception:
                    pass

    # Экспортируем локаторы как Dummy
    point_3d_list = calibrator.get_points_3d()
    names = list(locator_names)
    if len(names) < len(point_3d_list):
        names += [f"pt{i}" for i in range(len(names), len(point_3d_list))]
    for name, pt in zip(names, point_3d_list):
        pos = axes @ pt.reshape(3)
        dummy = rt.Dummy()
        dummy.name = name
        dummy.position = rt.Point3(*as_float_tuple(pos))


