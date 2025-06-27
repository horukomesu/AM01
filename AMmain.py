"""Main entry point for MyImageModelerPlugin.

Этот скрипт загружает Qt UI, созданный в ``AMUI.ui`` как QMainWindow, и вешает обработчики.
Используется ТОЛЬКО внутри Autodesk 3ds Max 2023+.
"""

from pathlib import Path
from PySide2 import QtWidgets, QtCore, QtGui, QtUiTools
import numpy as np
import importlib

import AMUtilities
importlib.reload(AMUtilities)


import AMCalibrationPlotter
importlib.reload(AMCalibrationPlotter)

import CameraCalibrator
importlib.reload(CameraCalibrator)
from CameraCalibrator import CameraCalibrator


# Держим ссылку глобально, чтобы окно не закрывалось сразу
main_window = None


class ImageViewer(QtWidgets.QGraphicsView):
    """Interactive viewer placed inside ``MainFrame``."""

    locator_added = QtCore.Signal(float, float)
    navigate = QtCore.Signal(int)  # emit +1/-1 to switch images

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap)
        self.mark_items = []

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

    def set_markers(self, markers, highlight_name=None):
        for item in self.mark_items:
            self.scene().removeItem(item)
        self.mark_items = []
        for m in markers:
            self._add_marker_item(
                m["x"], m["y"], m.get("name", ""), m.get("name") == highlight_name
            )

    def _add_marker_item(self, x_norm: float, y_norm: float, name: str, highlight: bool = False):
        rect = QtCore.QRectF(-5, -5, 10, 10)
        item = QtWidgets.QGraphicsEllipseItem(rect)
        color = QtCore.Qt.green if highlight else QtCore.Qt.red
        item.setBrush(QtGui.QBrush(color))
        item.setPen(QtGui.QPen(QtCore.Qt.black))
        pos = QtCore.QPointF(x_norm * self.sceneRect().width(), y_norm * self.sceneRect().height())
        item.setPos(pos)
        item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self.scene().addItem(item)
        text = QtWidgets.QGraphicsSimpleTextItem(name)
        text.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        text.setPos(pos + QtCore.QPointF(6, -6))
        self.scene().addItem(text)
        self.mark_items.append(item)
        self.mark_items.append(text)

    def mousePressEvent(self, event):
        if self.adding_locator:
            if event.button() == QtCore.Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                x_norm = pos.x() / self.sceneRect().width()
                y_norm = pos.y() / self.sceneRect().height()
                self.locator_added.emit(x_norm, y_norm)
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
            markers.append({"name": loc["name"], "x": pos["x"], "y": pos["y"]})
    return markers


def update_tree():
    """Rebuild the tree with image and locator nodes."""
    tree = main_window.MainTree
    tree.clear()

    img_root = QtWidgets.QTreeWidgetItem(tree, ["Images"])
    img_root.setData(0, QtCore.Qt.UserRole, ("images_root",))
    for idx, path in enumerate(main_window.image_paths):
        item = QtWidgets.QTreeWidgetItem(img_root, [Path(path).name])
        item.setData(0, QtCore.Qt.UserRole, ("image", idx))

    loc_root = QtWidgets.QTreeWidgetItem(tree, ["Locators"])
    loc_root.setData(0, QtCore.Qt.UserRole, ("loc_root",))
    for loc in getattr(main_window, "locators", []):
        item = QtWidgets.QTreeWidgetItem(loc_root, [loc["name"]])
        item.setData(0, QtCore.Qt.UserRole, ("locator", loc["name"]))

    img_root.setExpanded(True)
    loc_root.setExpanded(True)


def get_next_locator_name() -> str:
    """Return the minimal free locator name as ``locN``."""
    existing = {loc["name"] for loc in getattr(main_window, "locators", [])}
    idx = 1
    while f"loc{idx}" in existing:
        idx += 1
    return f"loc{idx}"


def exit_locator_mode():
    """Reset locator placement mode."""
    main_window.locator_mode = False
    main_window.viewer.adding_locator = False
    main_window.viewer.setCursor(QtCore.Qt.ArrowCursor)


def show_image(index: int, keep_view: bool = False):
    if 0 <= index < len(main_window.images):
        main_window.current_image_index = index
        main_window.viewer.load_image(main_window.images[index], keep_transform=keep_view)
        highlight = getattr(main_window, "selected_locator", None)
        main_window.viewer.set_markers(get_image_markers(index), highlight)


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
def new_scene():
    exit_locator_mode()
    main_window.image_paths = []
    main_window.images = []
    main_window.locators = []
    main_window.selected_locator = None
    update_tree()
    main_window.viewer._pixmap.setPixmap(QtGui.QPixmap())
    main_window.viewer.set_markers([])
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
    update_tree()
    if main_window.images:
        show_image(0)
    else:
        main_window.viewer._pixmap.setPixmap(QtGui.QPixmap())
        main_window.viewer.set_markers([])

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
        calibrator.load_locators(getattr(main_window, "locators", []))
        calibrator.detect_and_match_features()
        calibrator.run_reconstruction()
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            main_window, "Calibrate", f"Calibration failed:\n{exc}"
        )
        return

    main_window.calibration = calibrator
    error = calibrator.get_reprojection_error()
    msg = (
        f"Recovered {len(calibrator.get_camera_parameters())} cameras.\n"
        f"3D points: {len(calibrator.get_points_3d())}"
    )
    if error is not None:
        msg += f"\nReprojection error: {error:.4f}"

    QtWidgets.QMessageBox.information(main_window, "Calibration Completed", msg)




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
        AMUtilities.export_calibration_to_max(
            calibrator=main_window.calibration,
            image_paths=main_window.image_paths,
            locator_names=[f"pt{i}" for i in range(len(main_window.calibration.get_points_3d()))],
            sensor_width_mm=36.0  # или другой, если ты хочешь поддерживать crop
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
    QtWidgets.QMessageBox.information(main_window, "Worldspace", "Define worldspace not implemented.")

def define_reference_distance():
    QtWidgets.QMessageBox.information(main_window, "Reference Distance", "Define reference distance not implemented.")

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

        # Viewer setup inside MainFrame
        layout = QtWidgets.QVBoxLayout(main_window.MainFrame)
        layout.setContentsMargins(0, 0, 0, 0)
        main_window.viewer = ImageViewer(main_window.MainFrame)
        main_window.viewer.locator_added.connect(on_locator_added)
        main_window.viewer.navigate.connect(
            lambda step: (next_image() if step > 0 else prev_image())
        )
        layout.addWidget(main_window.viewer)

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

        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), main_window, next_image)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), main_window, prev_image)

        # Toolbar buttons
        main_window.btnAddLoc.clicked.connect(add_locator)
        main_window.btnCalibrate.clicked.connect(calibrate)
        main_window.btnDFWS.clicked.connect(define_worldspace)
        main_window.btnDFMM.clicked.connect(define_reference_distance)
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
