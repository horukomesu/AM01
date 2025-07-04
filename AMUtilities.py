"""Utility functions for MyImageModelerPlugin.

This module holds helper routines used across the application
such as loading images and reading/writing scene description files.

The functions are kept independent from any UI logic so they can be
tested in isolation.
"""

from __future__ import annotations
import sys
import os
from AMRZI_IO import read_rzi
from pathlib import Path
from typing import List, Dict, Any
import importlib

import numpy as np

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

# Добавить рядом с другими импортами
import Filesystem
importlib.reload(Filesystem)

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

def save_scene(image_paths: List[str], locators: List[Dict[str, Any]], path: str) -> None:
    import json
    data = {
        "format_version": FORMAT_VERSION,
        "images": image_paths,  # просто список имен, порядок важен
        "locators": locators,   # positions должны ссылаться по индексу
    }
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    Filesystem.save_ams(path, json_str, image_paths)


def load_scene_any(path: str) -> Dict[str, Any]:
    """
    Универсальная загрузка сцены (.ams или .rzi).

    Parameters
    ----------
    path : str
        Путь к сцене (расширение определяет тип).

    Returns
    -------
    Dict[str, Any] с ключами: "image_paths", "locators"
    """
    ext = Path(path).suffix.lower()
    if ext == ".rzi":
        return load_scene_rzi(path)
    elif ext == ".ams":
        return load_scene_ams(path)
    else:
        raise ValueError(f"Unsupported scene format: {ext}")


def load_scene_ams(path: str) -> Dict[str, Any]:
    import json
    raw = Filesystem.load_ams(path)
    data = json.loads(raw["scene"])

    if "images" not in data or "locators" not in data:
        raise ValueError("Scene missing 'images' or 'locators'")

    image_paths = list(data["images"])
    locators = list(data["locators"])

    for loc in locators:
        if "name" not in loc or "positions" not in loc:
            raise ValueError(f"Bad locator format: {loc}")
        if isinstance(loc["positions"], dict):
            loc["positions"] = {int(k): v for k, v in loc["positions"].items()}

    return {
        "image_paths": image_paths,
        "locators": locators,
    }



def load_scene_rzi(path: str) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Scene file not found: {path}")
    
    data = read_rzi(path)
    base_dir = Path(path).parent

    image_paths = []
    locators_map = {}

    for shot in data.get("shots", []):
        raw_path = shot.get("image_path", "") or shot.get("filename", "")
        try:
            img_path = Path(raw_path).expanduser().resolve(strict=False)
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
        "image_paths": image_paths,
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
