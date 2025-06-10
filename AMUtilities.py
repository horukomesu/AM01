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
