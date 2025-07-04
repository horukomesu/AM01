import os
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from PyQt6.QtGui import QImage, QColor

from AMRZI_IO import read_rzi  
import Filesystem 

FORMAT_VERSION = 1

def error_to_color(error: float, min_err: float = 0.0, max_err: float = 10.0) -> QColor:
    t = np.clip((error - min_err) / (max_err - min_err), 0.0, 1.0)
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    return QColor(r, g, 0)

def load_images(paths: List[str]) -> List[QImage]:
    images = []
    for p in paths:
        qimg = QImage(p)
        if not qimg.isNull():
            images.append(qimg)
    return images

def verify_paths(paths: List[str]) -> List[str]:
    return [p for p in paths if Path(p).exists()]

def save_scene(image_paths: List[str], locators: List[Dict[str, Any]], path: str) -> None:
    data = {
        "format_version": FORMAT_VERSION,
        "images": image_paths,
        "locators": locators,
    }
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    Filesystem.save_ams(path, json_str, image_paths)

def load_scene_any(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    if ext == ".rzi":
        return load_scene_rzi(path)
    elif ext == ".ams":
        return load_scene_ams(path)
    else:
        raise ValueError(f"Unsupported scene format: {ext}")

def load_scene_ams(path: str) -> Dict[str, Any]:
    raw = Filesystem.load_ams(path)

    scene_str = raw["scene"]
    if not isinstance(scene_str, str):
        raise TypeError(f"Expected 'scene' to be str, got {type(scene_str)}")

    data = json.loads(scene_str)

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
