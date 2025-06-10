"""Main entry point for MyImageModelerPlugin.

Этот скрипт загружает Qt UI, созданный в ``AMUI.ui`` как QMainWindow, и вешает обработчики.
Используется ТОЛЬКО внутри Autodesk 3ds Max 2023+.
"""

from pathlib import Path
from PySide2 import QtWidgets, QtCore, QtGui, QtUiTools

import AMUtilities
import AMCameraCalibrate
import numpy as np
import cv2

# Держим ссылку глобально, чтобы окно не закрывалось сразу
main_window = None


class ImageViewer(QtWidgets.QGraphicsView):
    """Interactive viewer placed inside ``MainFrame`` with modern controls."""

    locator_added = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap)
        self.mark_items = []

        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

        self._panning = False
        self._pan_start = QtCore.QPoint()
        self.adding_locator = False

    def load_image(self, qimg: QtGui.QImage):
        self.scene().setSceneRect(0, 0, qimg.width(), qimg.height())
        self._pixmap.setPixmap(QtGui.QPixmap.fromImage(qimg))
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
        if self.adding_locator and event.button() == QtCore.Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            x_norm = pos.x() / self.sceneRect().width()
            y_norm = pos.y() / self.sceneRect().height()
            self.locator_added.emit(x_norm, y_norm)
            return

        if event.button() == QtCore.Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == QtCore.Qt.RightButton:
            self._show_context_menu(event)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                next_image()
            else:
                prev_image()
            event.accept()
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        event.accept()

    def _show_context_menu(self, event):
        menu = QtWidgets.QMenu(self)
        menu.addAction("Add Locator", add_locator)
        menu.addAction("Calibrate", calibrate)
        menu.addAction("Load Images", import_images)
        menu.addAction("Save Scene", save_scene)
        menu.exec_(self.mapToGlobal(event.pos()))

    def mouseDoubleClickEvent(self, event):
        if self._pixmap.pixmap() and not self._pixmap.pixmap().isNull():
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)


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


def exit_locator_mode():
    """Reset locator placement mode."""
    main_window.locator_mode = False
    main_window.current_locator = None
    main_window.viewer.adding_locator = False
    main_window.viewer.setCursor(QtCore.Qt.ArrowCursor)


def show_image(index: int):
    if 0 <= index < len(main_window.images):
        main_window.current_image_index = index
        main_window.viewer.load_image(main_window.images[index])
        highlight = getattr(main_window, "selected_locator", None)
        main_window.viewer.set_markers(get_image_markers(index), highlight)


def next_image():
    idx = getattr(main_window, "current_image_index", 0) + 1
    if idx < len(main_window.images):
        show_image(idx)
        root = main_window.MainTree.topLevelItem(0)
        if root and idx < root.childCount():
            main_window.MainTree.setCurrentItem(root.child(idx))


def prev_image():
    idx = getattr(main_window, "current_image_index", 0) - 1
    if idx >= 0:
        show_image(idx)
        root = main_window.MainTree.topLevelItem(0)
        if root and idx < root.childCount():
            main_window.MainTree.setCurrentItem(root.child(idx))


def on_tree_selection_changed(current, _previous):
    if not current:
        return
    data = current.data(0, QtCore.Qt.UserRole)
    if not data:
        return
    if data[0] == "image":
        main_window.selected_locator = None
        show_image(data[1])
    elif data[0] == "locator":
        main_window.selected_locator = data[1]
        show_image(getattr(main_window, "current_image_index", 0))


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
    show_image(getattr(main_window, "current_image_index", 0))


def on_locator_added(x_norm: float, y_norm: float):
    idx = getattr(main_window, "current_image_index", 0)
    if idx >= len(main_window.image_paths):
        return
    loc = getattr(main_window, "current_locator", None)
    if not loc:
        return
    loc.setdefault("positions", {})[idx] = {"x": x_norm, "y": y_norm}
    update_tree()
    show_image(idx)



def new_scene():
    main_window.image_paths = []
    main_window.images = []
    main_window.locators = []
    main_window.current_locator = None
    main_window.selected_locator = None
    main_window.locator_counter = 1
    update_tree()
    main_window.viewer._pixmap.setPixmap(QtGui.QPixmap())
    main_window.viewer.set_markers([])
    QtWidgets.QMessageBox.information(main_window, "New", "Started a new scene.")

def import_images():
    paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
        main_window, "Select Images", "", "Images (*.png *.jpg *.jpeg *.tif)"
    )
    paths = AMUtilities.verify_paths(paths)
    main_window.image_paths = paths
    main_window.images = AMUtilities.load_images(paths)
    main_window.locators = []
    main_window.current_locator = None
    main_window.selected_locator = None
    main_window.locator_counter = 1
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
    scene = {
        "images": getattr(main_window, "image_paths", []),
        "locators": getattr(main_window, "locators", []),
    }
    AMUtilities.save_scene(scene, path)

def save_scene_as():
    save_scene()

def load_scene():
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        main_window, "Load Scene", "", "JSON (*.json)"
    )
    if not path:
        return
    scene = AMUtilities.load_scene(path)
    main_window.image_paths = scene.get("images", [])
    main_window.images = AMUtilities.load_images(main_window.image_paths)
    main_window.locators = scene.get("locators", [])
    main_window.current_locator = None
    main_window.selected_locator = None
    main_window.locator_counter = 1
    for loc in main_window.locators:
        try:
            num = int(loc["name"].lstrip("loc"))
            if num >= main_window.locator_counter:
                main_window.locator_counter = num + 1
        except Exception:
            pass
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
    if getattr(main_window, "locator_mode", False):
        exit_locator_mode()
    name = f"loc{main_window.locator_counter}"
    main_window.locator_counter += 1
    main_window.current_locator = {"name": name, "positions": {}}
    main_window.locators.append(main_window.current_locator)
    main_window.viewer.adding_locator = True
    main_window.viewer.setCursor(QtCore.Qt.CrossCursor)
    main_window.locator_mode = True
    QtWidgets.QMessageBox.information(
        main_window,
        "Add Locator",
        f"Placing {name}. Click on images to mark its position. Press Esc to finish.",
    )

def calibrate():
    if not getattr(main_window, "image_paths", []):
        QtWidgets.QMessageBox.warning(main_window, "Calibrate", "No images loaded for calibration.")
        return

    board_size = (9, 6)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    image_points = []
    object_points = []
    image_size = None

    for img_path in main_window.image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        ret, corners = cv2.findChessboardCorners(img, board_size, None)
        if ret:
            object_points.append(objp)
            image_points.append(corners)

    if not image_points:
        QtWidgets.QMessageBox.warning(main_window, "Calibrate", "Chessboard corners not found in any image.")
        return

    result = AMCameraCalibrate.calibrate_cameras(image_points, object_points, image_size)
    QtWidgets.QMessageBox.information(
        main_window,
        "Calibration Completed",
        f"Reprojection error: {result.reprojection_error:.4f}",
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
        main_window.current_locator = None
        main_window.selected_locator = None
        main_window.locator_mode = False
        main_window.locator_counter = 1

        # Viewer setup inside MainFrame
        layout = QtWidgets.QVBoxLayout(main_window.MainFrame)
        layout.setContentsMargins(0, 0, 0, 0)
        main_window.viewer = ImageViewer(main_window.MainFrame)
        main_window.viewer.locator_added.connect(on_locator_added)
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

        class _KeyFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() == QtCore.QEvent.KeyPress:
                    if event.key() == QtCore.Qt.Key_Escape:
                        exit_locator_mode()
                        return True
                    if event.key() == QtCore.Qt.Key_Right:
                        next_image()
                        return True
                    if event.key() == QtCore.Qt.Key_Left:
                        prev_image()
                        return True
                return super().eventFilter(obj, event)

        main_window._key_filter = _KeyFilter(main_window.viewer)
        main_window.viewer.installEventFilter(main_window._key_filter)
        main_window.MainTree.installEventFilter(main_window._key_filter)

        # Toolbar buttons
        main_window.btnAddLoc.clicked.connect(add_locator)
        main_window.btnCalibrate.clicked.connect(calibrate)
        main_window.btnDFWS.clicked.connect(define_worldspace)
        main_window.btnDFMM.clicked.connect(define_reference_distance)
        main_window.btnLocMod.clicked.connect(add_modeling_locator)

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
