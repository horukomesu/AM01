"""Main entry point for MyImageModelerPlugin.

Этот скрипт загружает Qt UI, созданный в ``AMUI.ui`` как QMainWindow, и вешает обработчики.
Используется ТОЛЬКО внутри Autodesk 3ds Max 2023+.
"""
import sys
import os
from pathlib import Path

try:
    from PySide2 import QtWidgets, QtCore, QtGui, QtUiTools
    QShortcutBase = QtWidgets.QShortcut
except ImportError:
    from PySide6 import QtWidgets, QtCore, QtGui, QtUiTools
    QShortcutBase = QtGui.QShortcut


from typing import Optional
import numpy as np
import importlib
import site
print(site.getusersitepackages())

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

import AMUtilities
importlib.reload(AMUtilities)
import GLSceneView
importlib.reload(GLSceneView)
from GLSceneView import GLSceneView


import CameraCalibrator
importlib.reload(CameraCalibrator)
from CameraCalibrator import CameraCalibrator


# Держим ссылку глобально, чтобы окно не закрывалось сразу
main_window = None


class ImageViewer(QtWidgets.QGraphicsView):
    """Interactive viewer placed inside ``MainFrame``."""

    locator_added = QtCore.Signal(float, float)
    navigate = QtCore.Signal(int)  # emit +1/-1 to switch images
    locator_clicked = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap)
        self.mark_items = []
        self.marker_info = []

        self.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)

        self._panning = False
        self._pan_start = QtCore.QPoint()
        self.adding_locator = False
        self.zoom_step = 1.2

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_action_menu)

        self._action_menu = None

    def load_image(self, qimg: QtGui.QImage, keep_transform: bool = False):
        current_transform = self.transform()
        h_val = self.horizontalScrollBar().value()
        v_val = self.verticalScrollBar().value()

        self.scene().setSceneRect(0, 0, qimg.width(), qimg.height())
        self._pixmap.setPixmap(QtGui.QPixmap.fromImage(qimg))

        if keep_transform:
            self.setTransform(current_transform)
            self.horizontalScrollBar().setValue(h_val)
            self.verticalScrollBar().setValue(v_val)
        else:
            self.resetTransform()
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def set_markers(self, markers, highlight_names=None, axis_lines=None, scale_line=None, origin_pos=None):
        highlight_names = set(highlight_names or [])
        for item in self.mark_items:
            self.scene().removeItem(item)
        self.mark_items = []
        self.marker_info = []
        for m in markers:
            self._add_marker_item(
                m["x"],
                m["y"],
                m.get("name", ""),
                m.get("name") in highlight_names,
                m.get("error"),
                QtGui.QColor("yellow") if m.get("name") in highlight_names else None
            )
            self.marker_info.append((m.get("name", ""), m["x"], m["y"]))

        if axis_lines:
            for line in axis_lines:
                p1 = line.get("p1")
                p2 = line.get("p2")
                color = line.get("color", QtGui.QColor("white"))
                pen = QtGui.QPen(color)
                pen.setWidth(2)
                if p1 and p2:
                    line_item = QtWidgets.QGraphicsLineItem(
                        p1[0] * self.sceneRect().width(),
                        p1[1] * self.sceneRect().height(),
                        p2[0] * self.sceneRect().width(),
                        p2[1] * self.sceneRect().height()
                    )
                    line_item.setPen(pen)
                    line_item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
                    self.scene().addItem(line_item)
                    self.mark_items.append(line_item)
                    mid = QtCore.QPointF(
                        (p1[0] + p2[0]) / 2 * self.sceneRect().width(),
                        (p1[1] + p2[1]) / 2 * self.sceneRect().height()
                    )
                else:
                    pt = p1 or p2
                    if pt is None:
                        continue
                    mid = QtCore.QPointF(
                        pt[0] * self.sceneRect().width(),
                        pt[1] * self.sceneRect().height()
                    )
                text = QtWidgets.QGraphicsSimpleTextItem(line.get("axis", ""))
                text.setBrush(QtGui.QBrush(color))
                text.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
                text.setPos(mid + QtCore.QPointF(6, -6))
                self.scene().addItem(text)
                self.mark_items.append(text)

        if scale_line and scale_line.get("p1") and scale_line.get("p2"):
            p1 = scale_line["p1"]
            p2 = scale_line["p2"]
            pen = QtGui.QPen(QtGui.QColor("yellow"))
            pen.setWidth(2)
            line_item = QtWidgets.QGraphicsLineItem(
                p1[0] * self.sceneRect().width(),
                p1[1] * self.sceneRect().height(),
                p2[0] * self.sceneRect().width(),
                p2[1] * self.sceneRect().height()
            )
            line_item.setPen(pen)
            line_item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            self.scene().addItem(line_item)
            self.mark_items.append(line_item)

        if origin_pos:
            pos = QtCore.QPointF(
                origin_pos[0] * self.sceneRect().width(),
                origin_pos[1] * self.sceneRect().height()
            )
            text = QtWidgets.QGraphicsSimpleTextItem("O")
            text.setBrush(QtGui.QBrush(QtGui.QColor("white")))
            text.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            text.setPos(pos + QtCore.QPointF(6, -6))
            self.scene().addItem(text)
            self.mark_items.append(text)

    def _add_marker_item(self, x_norm: float, y_norm: float, name: str, highlight: bool = False, error: float = 0.0, color: Optional[QtGui.QColor] = None):
        if color is None:
            color = AMUtilities.error_to_color(error or 0.0)
        pen = QtGui.QPen(color)
        pen.setWidth(2 if highlight else 1)
        pos = QtCore.QPointF(x_norm * self.sceneRect().width(), y_norm * self.sceneRect().height())

        line1 = QtWidgets.QGraphicsLineItem(-5, 0, 5, 0)
        line2 = QtWidgets.QGraphicsLineItem(0, -5, 0, 5)
        for line in (line1, line2):
            line.setPen(pen)
            line.setPos(pos)
            line.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            self.scene().addItem(line)
            self.mark_items.append(line)

        text = QtWidgets.QGraphicsSimpleTextItem(name)
        text.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        text.setPos(pos + QtCore.QPointF(6, -6))
        self.scene().addItem(text)
        self.mark_items.append(text)

    def _locator_at(self, x_norm: float, y_norm: float, thresh: float = 0.02):
        for name, x, y in self.marker_info:
            if ((x - x_norm) ** 2 + (y - y_norm) ** 2) ** 0.5 <= thresh:
                return name
        return None

    def mousePressEvent(self, event):
        if self.adding_locator:
            if event.button() == QtCore.Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                x_norm = pos.x() / self.sceneRect().width()
                y_norm = pos.y() / self.sceneRect().height()
                self.locator_added.emit(x_norm, y_norm)
                return

        if event.button() == QtCore.Qt.LeftButton:
            if any([
                getattr(main_window, "axis_definition_mode", False),
                getattr(main_window, "measurement_definition_mode", False),
                getattr(main_window, "origin_selection_mode", False),
            ]):
                pos = self.mapToScene(event.pos())
                x_norm = pos.x() / self.sceneRect().width()
                y_norm = pos.y() / self.sceneRect().height()
                name = self._locator_at(x_norm, y_norm)
                if name:
                    self.locator_clicked.emit(name)
                    return

        if event.button() == QtCore.Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.ControlModifier:
            step = 1 if event.angleDelta().y() > 0 else -1
            self.navigate.emit(step)
            return

        old_pos = self.mapToScene(event.pos())
        zoom_out = event.angleDelta().y() < 0
        factor = 1.0 / self.zoom_step if zoom_out else self.zoom_step
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def mouseDoubleClickEvent(self, event):
        if self._pixmap.pixmap() and not self._pixmap.pixmap().isNull():
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)

    def _show_action_menu(self, pos):
        if self._action_menu is None:
            self._action_menu = QtWidgets.QMenu(self)
            self._action_menu.addAction("Add Locator", add_locator)
            self._action_menu.addAction("Calibrate", calibrate)
            self._action_menu.addAction("Define Worldspace", define_worldspace)
            self._action_menu.addAction(
                "Define Reference Distance", define_reference_distance
            )
            self._action_menu.addAction(
                "Add Modeling Locator", add_modeling_locator
            )
        self._action_menu.exec_(self.mapToGlobal(pos))


def get_image_markers(index: int):
    """Return list of markers for the given image index."""
    markers = []
    for loc in getattr(main_window, "locators", []):
        pos = loc.get("positions", {}).get(index)
        if pos:
            markers.append({
                "name": loc["name"],
                "x": pos["x"],
                "y": pos["y"],
                "error": loc.get("error")
            })
    return markers


def update_tree():
    """Rebuild the tree with image and locator nodes."""
    tree = main_window.MainTree
    tree.clear()

    img_root = QtWidgets.QTreeWidgetItem(tree, ["Images"])
    img_root.setData(0, QtCore.Qt.UserRole, ("images_root",))
    img_errors = getattr(main_window, "image_errors", {})
    for idx, path in enumerate(main_window.image_paths):
        item = QtWidgets.QTreeWidgetItem(img_root, [Path(path).name])
        item.setData(0, QtCore.Qt.UserRole, ("image", idx))
        err = img_errors.get(idx)
        if err is not None:
            color = AMUtilities.error_to_color(err)
            icon = QtGui.QPixmap(16, 16)
            icon.fill(color)
            item.setIcon(0, QtGui.QIcon(icon))

    loc_root = QtWidgets.QTreeWidgetItem(tree, ["Locators"])
    loc_root.setData(0, QtCore.Qt.UserRole, ("loc_root",))
    for loc in getattr(main_window, "locators", []):
        item = QtWidgets.QTreeWidgetItem(loc_root, [loc["name"]])
        item.setData(0, QtCore.Qt.UserRole, ("locator", loc["name"]))
        color = AMUtilities.error_to_color(loc.get("error", 0.0))
        icon = QtGui.QPixmap(16, 16)
        icon.fill(color)
        item.setIcon(0, QtGui.QIcon(icon))

    img_root.setExpanded(True)
    loc_root.setExpanded(True)


def get_next_locator_name() -> str:
    """Return the minimal free locator name as ``locN``."""
    existing = {loc["name"] for loc in getattr(main_window, "locators", [])}
    idx = 1
    while f"loc{idx}" in existing:
        idx += 1
    return f"loc{idx}"


def _find_locator(name: str):
    for loc in getattr(main_window, "locators", []):
        if loc["name"] == name:
            return loc
    return None


def exit_locator_mode():
    """Reset locator placement mode."""
    main_window.locator_mode = False
    main_window.viewer.adding_locator = False
    main_window.viewer.setCursor(QtCore.Qt.ArrowCursor)


def show_image(index: int, keep_view: bool = False):
    if 0 <= index < len(main_window.images):
        main_window.current_image_index = index
        main_window.viewer.load_image(main_window.images[index], keep_transform=keep_view)
        highlight_names = set()
        if getattr(main_window, "selected_locator", None):
            highlight_names.add(main_window.selected_locator)
        if getattr(main_window, "scale_pair", None):
            highlight_names.update(main_window.scale_pair)

        axis_lines = []
        colors = {"X": QtGui.QColor("red"), "Y": QtGui.QColor("green"), "Z": QtGui.QColor("blue")}
        for axis, pair in getattr(main_window, "axis_points", {}).items():
            if len(pair) != 2:
                continue
            loc1 = _find_locator(pair[0])
            loc2 = _find_locator(pair[1])
            p1 = loc1.get("positions", {}).get(index) if loc1 else None
            p2 = loc2.get("positions", {}).get(index) if loc2 else None
            if not p1 and not p2:
                continue
            axis_lines.append({
                "axis": axis,
                "p1": (p1["x"], p1["y"]) if p1 else None,
                "p2": (p2["x"], p2["y"]) if p2 else None,
                "color": colors.get(axis, QtGui.QColor("white"))
            })

        scale_line = None
        if getattr(main_window, "scale_pair", None):
            loc1 = _find_locator(main_window.scale_pair[0])
            loc2 = _find_locator(main_window.scale_pair[1])
            p1 = loc1.get("positions", {}).get(index) if loc1 else None
            p2 = loc2.get("positions", {}).get(index) if loc2 else None
            if p1 or p2:
                scale_line = {
                    "names": main_window.scale_pair,
                    "p1": (p1["x"], p1["y"]) if p1 else None,
                    "p2": (p2["x"], p2["y"]) if p2 else None,
                }

        origin_pos = None
        if getattr(main_window, "origin_locator_name", None):
            loc = _find_locator(main_window.origin_locator_name)
            pos = loc.get("positions", {}).get(index) if loc else None
            if pos:
                origin_pos = (pos["x"], pos["y"])

        main_window.viewer.set_markers(
            get_image_markers(index),
            highlight_names=highlight_names,
            axis_lines=axis_lines,
            scale_line=scale_line,
            origin_pos=origin_pos,
        )
        if hasattr(main_window, "viewer3D") and main_window.viewer3D.isVisible():
            main_window.viewer3D.set_active_camera(index)
            main_window.viewer3D.set_overlay_data(
                axis_points=getattr(main_window, "axis_points", {}),
                scale_pair=getattr(main_window, "scale_pair", None),
                origin_name=getattr(main_window, "origin_locator_name", None),
            )


def next_image():
    """Switch to the next image cyclically while keeping the current view."""
    if not main_window.images:
        return
    idx = (getattr(main_window, "current_image_index", 0) + 1) % len(main_window.images)
    show_image(idx, keep_view=True)


def prev_image():
    """Switch to the previous image cyclically while keeping the current view."""
    if not main_window.images:
        return
    idx = (getattr(main_window, "current_image_index", 0) - 1) % len(main_window.images)
    show_image(idx, keep_view=True)


def on_tree_selection_changed(current, _previous):
    if not current:
        return
    data = current.data(0, QtCore.Qt.UserRole)
    if not data:
        return
    if data[0] == "image":
        main_window.selected_locator = None
        exit_locator_mode()
        show_image(data[1])
        if hasattr(main_window, "viewer3D") and main_window.viewer3D.isVisible():
            main_window.viewer3D.set_active_camera(data[1])
    elif data[0] == "locator":
        exit_locator_mode()
        main_window.selected_locator = data[1]
        main_window.viewer.adding_locator = True
        main_window.viewer.setCursor(QtCore.Qt.CrossCursor)
        main_window.locator_mode = True
        show_image(getattr(main_window, "current_image_index", 0), keep_view=True)


def delete_selected_locator():
    item = main_window.MainTree.currentItem()
    if not item:
        return
    data = item.data(0, QtCore.Qt.UserRole)
    if not data or data[0] != "locator":
        return
    name = data[1]
    main_window.locators = [l for l in main_window.locators if l["name"] != name]
    if main_window.selected_locator == name:
        main_window.selected_locator = None
    update_tree()
    show_image(getattr(main_window, "current_image_index", 0), keep_view=True)


def on_locator_added(x_norm: float, y_norm: float):
    idx = getattr(main_window, "current_image_index", 0)
    if idx >= len(main_window.image_paths):
        return
    name = getattr(main_window, "selected_locator", None)
    if not name:
        return
    loc = next((l for l in main_window.locators if l["name"] == name), None)
    if not loc:
        return
    loc.setdefault("positions", {})[idx] = {"x": x_norm, "y": y_norm}
    update_tree()
    show_image(idx, keep_view=True)


def on_locator_clicked(name: str):
    if getattr(main_window, "axis_definition_mode", False):
        axis = getattr(main_window, "_current_axis", "X")
        main_window.axis_points.setdefault(axis, [])
        if len(main_window.axis_points[axis]) < 2:
            main_window.axis_points[axis].append(name)
            if len(main_window.axis_points[axis]) == 2:
                if axis == "X":
                    main_window._current_axis = "Y"
                elif axis == "Y":
                    main_window._current_axis = "Z"
                else:
                    main_window.axis_definition_mode = False
                    main_window._current_axis = None
        show_image(getattr(main_window, "current_image_index", 0), keep_view=True)
        if not main_window.axis_definition_mode and hasattr(main_window, "viewer3D"):
            main_window.viewer3D.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
    elif getattr(main_window, "measurement_definition_mode", False):
        tmp = getattr(main_window, "_scale_temp", [])
        if len(tmp) < 2:
            tmp.append(name)
            main_window._scale_temp = tmp
        if len(tmp) == 2:
            dist, ok = QtWidgets.QInputDialog.getDouble(
                main_window, "Reference Distance", "distance (m):", decimals=3
            )
            if ok:
                main_window.known_distance = float(dist)
                main_window.scale_pair = tuple(tmp)
        main_window._scale_temp = []
        main_window.measurement_definition_mode = False
        show_image(getattr(main_window, "current_image_index", 0), keep_view=True)
        if hasattr(main_window, "viewer3D"):
            main_window.viewer3D.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
    elif getattr(main_window, "origin_selection_mode", False):
        main_window.origin_locator_name = name
        main_window.origin_selection_mode = False
        show_image(getattr(main_window, "current_image_index", 0), keep_view=True)
        if hasattr(main_window, "viewer3D"):
            main_window.viewer3D.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
def new_scene():
    exit_locator_mode()
    main_window.image_paths = []
    main_window.images = []
    main_window.locators = []
    main_window.selected_locator = None
    main_window.image_errors = {}
    update_tree()
    main_window.viewer._pixmap.setPixmap(QtGui.QPixmap())
    main_window.viewer.set_markers([])
    if hasattr(main_window, "viewer3D"):
        main_window.viewer3D.texture_ids.clear()
        main_window.viewer3D.hide()
    QtWidgets.QMessageBox.information(main_window, "New", "Started a new scene.")

def import_images():
    exit_locator_mode()
    paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
        main_window, "Select Images", "", "Images (*.png *.jpg *.jpeg *.tif)"
    )
    paths = AMUtilities.verify_paths(paths)
    main_window.image_paths = paths
    main_window.images = AMUtilities.load_images(paths)
    main_window.locators = []
    main_window.selected_locator = None
    main_window.image_errors = {}
    if hasattr(main_window, "viewer3D"):
        main_window.viewer3D.load_images(main_window.images)
    update_tree()
    if main_window.images:
        show_image(0)
    else:
        main_window.viewer._pixmap.setPixmap(QtGui.QPixmap())
        main_window.viewer.set_markers([])
        if hasattr(main_window, "viewer3D"):
            main_window.viewer3D.hide()

def save_scene():
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        main_window, "Save Scene", "", "JSON (*.json)"
    )
    if not path:
        return
    AMUtilities.save_scene(
        main_window.image_paths,
        main_window.locators,
        path
    )


def save_scene_as():
    save_scene()

def load_scene():
    exit_locator_mode()
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        main_window, "Load Scene", "", "JSON (*.json)"
    )
    if not path:
        return
    scene = AMUtilities.load_scene(path)
    main_window.image_paths = scene["images"]
    main_window.images = AMUtilities.load_images(main_window.image_paths)
    main_window.locators = scene["locators"]
    main_window.selected_locator = None
    main_window.image_errors = {}
    if hasattr(main_window, "viewer3D"):
        main_window.viewer3D.load_images(main_window.images)
    update_tree()
    if main_window.images:
        show_image(0)
    else:
        main_window.viewer._pixmap.setPixmap(QtGui.QPixmap())
        main_window.viewer.set_markers([])


def open_recent_project():
    QtWidgets.QMessageBox.information(main_window, "Recent", "Recent projects not implemented.")

def preferences():
    QtWidgets.QMessageBox.information(main_window, "Preferences", "Preferences dialog not implemented.")

def undo():
    QtWidgets.QMessageBox.information(main_window, "Undo", "Undo not implemented.")

def redo():
    QtWidgets.QMessageBox.information(main_window, "Redo", "Redo not implemented.")

def add_locator():
    if not main_window.images:
        QtWidgets.QMessageBox.warning(main_window, "Add Locator", "No image loaded.")
        return
    exit_locator_mode()
    name = get_next_locator_name()
    loc = {"name": name, "positions": {}}
    main_window.locators.append(loc)
    update_tree()
    main_window.selected_locator = name
    main_window.viewer.adding_locator = True
    main_window.locator_mode = True
    main_window.viewer.setCursor(QtCore.Qt.CrossCursor)
    # select item in tree
    items = main_window.MainTree.findItems(name, QtCore.Qt.MatchRecursive)
    if items:
        main_window.MainTree.setCurrentItem(items[0])


def calibrate():
    """Calibrate cameras using manually placed locators."""
    if not getattr(main_window, "image_paths", []):
        QtWidgets.QMessageBox.warning(
            main_window, "Calibrate", "No images loaded for calibration."
        )
        return

    calibrator = CameraCalibrator()
    try:
        calibrator.load_images(main_window.image_paths)

        # ---- Новое: собрать point_data из locators ----
        point_data = {}
        for set_id, loc in enumerate(getattr(main_window, "locators", [])):
            for img_idx, pos in loc.get("positions", {}).items():
                if set_id not in point_data:
                    point_data[set_id] = {}
                # Преобразовать к целочисленному индексу, если ключ-строка
                try:
                    img_idx_int = int(img_idx)
                except Exception:
                    img_idx_int = img_idx
                point_data[set_id][img_idx_int] = (pos["x"] * main_window.images[img_idx_int].width(), 
                                                   pos["y"] * main_window.images[img_idx_int].height())
        calibrator.load_point_data(point_data)
        success = calibrator.calibrate()
        if not success:
            QtWidgets.QMessageBox.critical(
                main_window, "Calibrate", f"Calibration failed. Check your points."
            )
            return
    except Exception as exc:
        import traceback
        QtWidgets.QMessageBox.critical(
            main_window, "Calibrate", f"Calibration failed:\n{exc}\n{traceback.format_exc()}"
        )
        return

    main_window.calibration = calibrator
    err_dict = calibrator.get_reprojection_error() or {}
    for idx, loc in enumerate(getattr(main_window, "locators", [])):
        loc["error"] = err_dict.get(str(idx), None)
    main_window.image_errors = calibrator.get_reprojection_errors_per_image() or {}

    msg = (
        f"Recovered {len(calibrator.get_camera_parameters())} cameras.\n"
        f"3D points: {len(calibrator.get_points_3d())}"
    )
    if err_dict:
        avg_err = sum(err_dict.values()) / len(err_dict)
        msg += f"\nReprojection error: {avg_err:.4f}"
    if main_window.image_errors:
        avg_img_err = sum(main_window.image_errors.values()) / len(main_window.image_errors)
        msg += f"\nImage error: {avg_img_err:.4f}"

    QtWidgets.QMessageBox.information(main_window, "Calibration Completed", msg)
    update_tree()
    show_image(getattr(main_window, "current_image_index", 0), keep_view=True)
    if hasattr(main_window, "viewer3D"):
        names = [loc["name"] for loc in main_window.locators]
        main_window.viewer3D.set_scene(calibrator, err_dict, names)
        if main_window.images:
            main_window.viewer3D.image_width = main_window.images[0].width()
        main_window.viewer3D.set_active_camera(getattr(main_window, "current_image_index", 0))
        main_window.viewer3D.set_overlay_data(
            axis_points=getattr(main_window, "axis_points", {}),
            scale_pair=getattr(main_window, "scale_pair", None),
            origin_name=getattr(main_window, "origin_locator_name", None),
        )
        main_window.viewer3D.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        main_window.viewer3D.show()


def compute_worldspace_transform(calibrator: CameraCalibrator):
    axes = getattr(main_window, "axis_points", {})
    if not calibrator.calibration_results or not axes:
        return np.eye(3), np.zeros(3), 1.0

    def loc_3d(name: str):
        for idx, loc in enumerate(getattr(main_window, "locators", [])):
            if loc["name"] == name:
                pts = calibrator.get_points_3d()
                if idx < len(pts):
                    return np.array(pts[idx], dtype=float)
        return None

    pts = {}
    for axis, pair in axes.items():
        if len(pair) != 2:
            return np.eye(3), np.zeros(3), 1.0
        p1 = loc_3d(pair[0])
        p2 = loc_3d(pair[1])
        if p1 is None or p2 is None:
            return np.eye(3), np.zeros(3), 1.0
        pts[axis] = (p1, p2)

    x = pts.get("X")
    y = pts.get("Y")
    z = pts.get("Z")
    if not (x and y and z):
        return np.eye(3), np.zeros(3), 1.0

    def norm(v):
        n = np.linalg.norm(v)
        return v / n if n else v

    Rx = norm(x[1] - x[0])
    Ry_temp = y[1] - y[0]
    Ry = norm(Ry_temp - np.dot(Ry_temp, Rx) * Rx)
    Rz_temp = z[1] - z[0]
    Rz = norm(Rz_temp - np.dot(Rz_temp, Rx) * Rx - np.dot(Rz_temp, Ry) * Ry)
    R = np.column_stack((Rx, Ry, Rz))

    origin = np.zeros(3)
    if getattr(main_window, "origin_locator_name", None):
        o = loc_3d(main_window.origin_locator_name)
        if o is not None:
            origin = o

    scale = 1.0
    if getattr(main_window, "scale_pair", None) and main_window.known_distance:
        p1 = loc_3d(main_window.scale_pair[0])
        p2 = loc_3d(main_window.scale_pair[1])
        if p1 is not None and p2 is not None:
            measured = np.linalg.norm(p1 - p2)
            if measured > 1e-8:
                scale = float(main_window.known_distance) / measured

    return R, origin, scale





def move_to_scene():
    if not hasattr(main_window, "calibration") or main_window.calibration is None:
        QtWidgets.QMessageBox.warning(
            main_window, "Move to Scene", "No calibration result found. Run calibration first."
        )
        return
    if not getattr(main_window, "image_paths", []):
        QtWidgets.QMessageBox.warning(
            main_window, "Move to Scene", "No images loaded."
        )
        return
    try:
        R, origin, scale = compute_worldspace_transform(main_window.calibration)
        AMUtilities.export_calibration_to_max(
            calibrator=main_window.calibration,
            image_paths=main_window.image_paths,
            locator_names=[f"pt{i}" for i in range(len(main_window.calibration.get_points_3d()))],
            world_R=R,
            world_origin=origin,
            world_scale=scale,
        )
        QtWidgets.QMessageBox.information(
            main_window, "Move to Scene", "Exported cameras and locators to 3ds Max scene."
        )
    except Exception as exc:
        import traceback
        QtWidgets.QMessageBox.critical(
            main_window, "Move to Scene",
            f"Export failed:\n{exc}\n{traceback.format_exc()}"
        )




def define_worldspace():
    exit_locator_mode()
    main_window.axis_definition_mode = True
    main_window.measurement_definition_mode = False
    main_window.origin_selection_mode = False
    main_window.axis_points = {"X": [], "Y": [], "Z": []}
    main_window._current_axis = "X"
    if hasattr(main_window, "viewer3D"):
        main_window.viewer3D.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
    QtWidgets.QMessageBox.information(
        main_window,
        "Worldspace",
        "Select two locators for X, then Y and Z axes."
    )

def define_reference_distance():
    exit_locator_mode()
    main_window.axis_definition_mode = False
    main_window.measurement_definition_mode = True
    main_window.origin_selection_mode = False
    main_window._scale_temp = []
    if hasattr(main_window, "viewer3D"):
        main_window.viewer3D.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
    QtWidgets.QMessageBox.information(
        main_window,
        "Reference Distance",
        "Select two locators to define known distance."
    )

def set_center():
    exit_locator_mode()
    main_window.axis_definition_mode = False
    main_window.measurement_definition_mode = False
    main_window.origin_selection_mode = True
    if hasattr(main_window, "viewer3D"):
        main_window.viewer3D.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
    QtWidgets.QMessageBox.information(
        main_window,
        "Set Center",
        "Click a locator to set as scene origin."
    )

def add_modeling_locator():
    QtWidgets.QMessageBox.information(main_window, "Modeling Locator", "Add modeling locator not implemented.")

# -- Автозапуск окна при загрузке скрипта --
try:
    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication не найден. Скрипт должен запускаться внутри 3ds Max!")

    if main_window is None:
        ui_path = Path(__file__).with_name("AMUI.ui")
        loader = QtUiTools.QUiLoader()
        ui_file = QtCore.QFile(str(ui_path))
        ui_file.open(QtCore.QFile.ReadOnly)
        main_window = loader.load(ui_file, None)
        ui_file.close()

        # Инициализация служебных полей для сцены и изображений
        main_window.image_paths = []
        main_window.images = []
        main_window.locators = []
        main_window.selected_locator = None
        main_window.locator_mode = False
        main_window.image_errors = {}
        main_window.axis_points = {"X": [], "Y": [], "Z": []}
        main_window.scale_pair = None
        main_window.known_distance = None
        main_window.origin_locator_name = None
        main_window.axis_definition_mode = False
        main_window.measurement_definition_mode = False
        main_window.origin_selection_mode = False
        main_window._current_axis = None
        main_window._scale_temp = []

        # Viewer setup inside MainFrame
        layout = QtWidgets.QHBoxLayout(main_window.MainFrame)
        layout.setContentsMargins(0, 0, 0, 0)
        main_window.viewer = ImageViewer(main_window.MainFrame)
        main_window.viewer3D = GLSceneView(main_window.MainFrame)

        main_window.viewer.locator_added.connect(on_locator_added)
        main_window.viewer.navigate.connect(
            lambda step: (next_image() if step > 0 else prev_image())
        )
        main_window.viewer3D.locator_clicked.connect(on_locator_clicked)

        layout.addWidget(main_window.viewer)
        main_window.viewer3D.setGeometry(main_window.viewer.geometry())
        main_window.viewer3D.raise_()
        main_window.viewer3D.hide()

        class _OverlayUpdater(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.Resize:
                    main_window.viewer3D.setGeometry(main_window.viewer.geometry())
                return super().eventFilter(obj, event)

        main_window._overlay_updater = _OverlayUpdater(main_window.MainFrame)
        main_window.MainFrame.installEventFilter(main_window._overlay_updater)
        main_window.viewer.installEventFilter(main_window._overlay_updater)

        # Tree selection and delete key
        main_window.MainTree.currentItemChanged.connect(on_tree_selection_changed)

        class _DeleteFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Delete:
                    delete_selected_locator()
                    return True
                return super().eventFilter(obj, event)

        main_window._del_filter = _DeleteFilter(main_window.MainTree)
        main_window.MainTree.installEventFilter(main_window._del_filter)

        class _EscFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Escape:
                    exit_locator_mode()
                    return True
                return super().eventFilter(obj, event)

        main_window._esc_filter = _EscFilter(main_window.viewer)
        main_window.viewer.installEventFilter(main_window._esc_filter)

        QShortcutBase(QtGui.QKeySequence(QtCore.Qt.Key_Right), main_window).activated.connect(next_image)
        QShortcutBase(QtGui.QKeySequence(QtCore.Qt.Key_Left), main_window).activated.connect(prev_image)



        # Toolbar buttons
        main_window.btnAddLoc.clicked.connect(add_locator)
        main_window.btnCalibrate.clicked.connect(calibrate)
        main_window.btnDFWS.clicked.connect(define_worldspace)
        main_window.btnDFMM.clicked.connect(define_reference_distance)
        main_window.btnSetCenter.clicked.connect(set_center)
        main_window.btnLocMod.clicked.connect(add_modeling_locator)
        main_window.btnMoveToScene.clicked.connect(move_to_scene)

        # Menu actions
        main_window.actionNEW.triggered.connect(new_scene)
        main_window.actionOpen.triggered.connect(load_scene)
        main_window.actionSave.triggered.connect(save_scene)
        main_window.actionSave_As.triggered.connect(save_scene_as)
        main_window.actionLoad.triggered.connect(import_images)
        main_window.actionRecent_Projects.triggered.connect(open_recent_project)
        main_window.actionPreferences.triggered.connect(preferences)
        main_window.actionUndo.triggered.connect(undo)
        main_window.actionRedo.triggered.connect(redo)

        main_window.setWindowTitle("AutoModeler")
        main_window.resize(1200, 900)
        main_window.show()

except Exception as e:
    import traceback
    QtWidgets.QMessageBox.critical(None, "AutoModeler Error", f"Ошибка запуска:\n{traceback.format_exc()}")
