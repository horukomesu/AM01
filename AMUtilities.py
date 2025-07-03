"""Utility functions for MyImageModelerPlugin.

This module holds helper routines used across the application
such as loading images and reading/writing scene description files.

The functions are kept independent from any UI logic so they can be
tested in isolation.
"""

from __future__ import annotations
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import importlib

import numpy as np

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

# Добавить рядом с другими импортами
import AMRZI_IO
importlib.reload(AMRZI_IO)
from AMRZI_IO import read_rzi, write_rzi

from CameraCalibrator import CameraCalibrator


try:
    from PySide2 import QtGui
except ImportError:
    from PySide6 import QtGui

FORMAT_VERSION = 1

def error_to_color(error: float, min_err: float = 0.0, max_err: float = 10.0) -> QtGui.QColor:
    """Convert a numeric error value to a QColor along a green-red gradient."""
    t = np.clip((error - min_err) / (max_err - min_err), 0.0, 1.0)
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    return QtGui.QColor(r, g, 0)

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
    """Сохраняет текущую сцену в .rzi (RZML-файл, совместимый с ImageModeler)."""
    # Загружаем изображения, чтобы узнать разрешения
    images = load_images(image_paths)
    if not images:
        raise RuntimeError("Нет доступных изображений для сохранения сцены.")

    width = images[0].width()
    height = images[0].height()

    # Примерный FOV по горизонтали — позже можно улучшить через CameraCalibrator
    default_fovx = 60.0

    rzi_data = {
        "version": "1.4.3",
        "path": str(Path(path).resolve()),
        "locators": [],
        "cameras": [{
            "id": 1,
            "name": "Camera Device",
            "width": width,
            "height": height,
            "sensor_width": 42.6667,  # по умолчанию (можно сделать опцией)
            "fovx": default_fovx,
            "distortion_type": "disto3i",
        }],
        "shots": []
    }

    for idx, img_path in enumerate(image_paths):
        if idx >= len(images):  # safety
            continue

        shot = {
            "id": idx + 1,
            "filename": Path(img_path).name,
            "camera_id": 1,
            "width": images[idx].width(),
            "height": images[idx].height(),
            "fovx": default_fovx,
            "rotation": {"x": -180},
            "image_path": str(Path(img_path).resolve()),
            "markers": []
        }

        for loc_idx, loc in enumerate(locators):
            if "positions" in loc and idx in loc["positions"]:
                pos = loc["positions"][idx]
                shot["markers"].append({
                    "locator_id": loc_idx + 1,
                    "x": float(pos["x"]),
                    "y": float(pos["y"]),
                })

        rzi_data["shots"].append(shot)

    for loc_idx, loc in enumerate(locators):
        rzi_data["locators"].append({
            "id": loc_idx + 1,
            "name": loc.get("name", f"loc{loc_idx+1}"),
        })

    write_rzi(path, rzi_data)



def _clean_path(p: str) -> Path:
    # Убираем начальные двойные слэши, заменяем на правильный путь
    p = p.lstrip("/\\")  # удаляет начальные \ или /
    return Path(p).expanduser()



def load_scene(path: str) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Scene file not found: {path}")
    
    data = read_rzi(path)
    base_dir = Path(path).parent

    image_paths = []
    locators_map = {}

    for shot in data.get("shots", []):
        raw_path = shot.get("image_path", "") or shot.get("filename", "")
        try:
            img_path = _clean_path(raw_path).resolve(strict=False)
        except Exception:
            img_path = base_dir / shot.get("filename", "")
        if not img_path.exists():
            fallback = base_dir / shot.get("filename", "")
            if fallback.exists():
                img_path = fallback
        image_paths.append(str(img_path))

    for shot_idx, shot in enumerate(data.get("shots", [])):
        for m in shot.get("markers", []):
            lid = m["locator_id"]
            if lid not in locators_map:
                name = next((l["name"] for l in data.get("locators", []) if l["id"] == lid), f"loc{lid}")
                locators_map[lid] = {"name": name, "positions": {}}
            locators_map[lid]["positions"][shot_idx] = {
                "x": m["x"],
                "y": m["y"]
            }

    return {
        "images": image_paths,
        "locators": list(locators_map.values()),
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
