"""Чтение и запись RZML-файлов Autodesk ImageModeler.

Модуль предоставляет функции `read_rzi` и `write_rzi` для работы с
форматом .rzi (RZML). Формат представляет собой XML-документ,
используемый ImageModeler 2009 для хранения сцен с камерами и
локаторами.
"""

from __future__ import annotations

from typing import Any, Dict, List
import xml.etree.ElementTree as ET


def parse_locators(root: ET.Element) -> List[Dict[str, Any]]:
    """Разбирает элементы <L> и возвращает список локаторов."""
    locators = []
    for l_el in root.findall("L"):
        try:
            loc_id = int(l_el.get("i", "0"))
        except ValueError:
            continue
        locators.append({
            "id": loc_id,
            "name": l_el.get("n", ""),
        })
    return locators


def parse_cameras(root: ET.Element) -> List[Dict[str, Any]]:
    """Разбирает элементы <CINF> со сведениями о камерах."""
    cameras = []
    for c_el in root.findall("CINF"):
        try:
            cam_id = int(c_el.get("i", "0"))
        except ValueError:
            continue
        cameras.append({
            "id": cam_id,
            "name": c_el.get("n", ""),
            "width": int(float(c_el.get("sw", "0"))),
            "height": int(float(c_el.get("sh", "0"))),
            "sensor_width": float(c_el.get("fbw", "0")),
            "fovx": float(c_el.get("fovx", "0")),
            "distortion_type": c_el.get("distoType", ""),
        })
    return cameras


def _parse_marker(m_el: ET.Element) -> Dict[str, Any]:
    """Возвращает информацию о маркере локатора."""
    try:
        locator_id = int(m_el.get("i", "0"))
    except ValueError:
        locator_id = 0
    x = float(m_el.get("x", "0")) / 100.0
    y = float(m_el.get("y", "0")) / 100.0
    return {
        "locator_id": locator_id,
        "x": x,
        "y": y,
    }


def parse_shots(root: ET.Element) -> List[Dict[str, Any]]:
    """Разбор кадров <SHOT> со всеми вложенными данными."""
    shots = []
    for s_el in root.findall("SHOT"):
        try:
            shot_id = int(s_el.get("i", "0"))
        except ValueError:
            continue
        shot = {
            "id": shot_id,
            "filename": s_el.get("n", ""),
            "camera_id": int(float(s_el.get("ci", "0"))),
            "width": int(float(s_el.get("w", "0"))),
            "height": int(float(s_el.get("h", "0"))),
        }

        cfrm = s_el.find("CFRM")
        if cfrm is not None:
            shot["fovx"] = float(cfrm.get("fovx", "0"))
            rot_el = cfrm.find("R")
            if rot_el is not None:
                rotation = {}
                for ax in ("x", "y", "z"):
                    if ax in rot_el.attrib:
                        rotation[ax] = float(rot_el.get(ax))
                if rotation:
                    shot["rotation"] = rotation
        ipln = s_el.find("IPLN")
        if ipln is not None:
            shot["image_path"] = ipln.get("img", "")
            markers = []
            ifrm = ipln.find("IFRM")
            if ifrm is not None:
                for m in ifrm.findall("M"):
                    markers.append(_parse_marker(m))
            shot["markers"] = markers
        shots.append(shot)
    return shots


def read_rzi(path: str) -> Dict[str, Any]:
    """Загружает RZML-файл и возвращает словарь со сценой."""
    tree = ET.parse(path)
    root = tree.getroot()

    data = {
        "version": root.get("v", ""),
        "path": root.get("path", ""),
        "locators": parse_locators(root),
        "cameras": parse_cameras(root),
        "shots": parse_shots(root),
    }
    return data




