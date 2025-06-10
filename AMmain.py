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
    """Interactive viewer placed inside ``MainFrame``."""

    locator_added = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap)
        self.mark_items = []

        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

        self._panning = False
        self._pan_start = QtCore.QPoint()
        self.adding_locator = False

        # Zoom behaviour configuration
        self.zoom_step = 1.2
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_action_menu)

    def _show_action_menu(self, pos):
        if main_window is None:
            return
        menu = getattr(main_window, "action_menu", None)
        if menu:
            menu.exec_(self.mapToGlobal(pos))

    def load_image(self, qimg: QtGui.QImage):
        self.scene().setSceneRect(0, 0, qimg.width(), qimg.height())
        self._pixmap.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def set_markers(self, markers):
        for item in self.mark_items:
            self.scene().removeItem(item)
        self.mark_items = []
        for m in markers:
            self._add_marker_item(m["x"], m["y"], m.get("name", ""))

    def _add_marker_item(self, x_norm: float, y_norm: float, name: str):
        rect = QtCore.QRectF(-5, -5, 10, 10)
        item = QtWidgets.QGraphicsEllipseItem(rect)
        item.setBrush(QtGui.QBrush(QtCore.Qt.red))
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
            self.adding_locator = False
            return

        if event.button() == QtCore.Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            return

        if event.button() == QtCore.Qt.RightButton:
            self._show_action_menu(event.pos())
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if event.modifiers() & QtCore.Qt.ControlModifier:
            if delta < 0:
                show_image(getattr(main_window, "current_image_index", 0) + 1)
            elif delta > 0:
                show_image(getattr(main_window, "current_image_index", 0) - 1)
            return

        factor = self.zoom_step if delta > 0 else 1.0 / self.zoom_step
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        if self._pixmap.pixmap() and not self._pixmap.pixmap().isNull():
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)


def update_tree():
    """Rebuild the scene tree from ``main_window.scene``."""
    main_window.MainTree.clear()
    for idx, path in enumerate(main_window.image_paths):
        img_item = QtWidgets.QTreeWidgetItem(main_window.MainTree, [Path(path).name])
        img_item.setData(0, QtCore.Qt.UserRole, ("image", idx))
        markers = main_window.scene[idx]["markers"]
        for m_idx, marker in enumerate(markers):
            loc_item = QtWidgets.QTreeWidgetItem(img_item, [marker["name"]])
            loc_item.setData(0, QtCore.Qt.UserRole, ("locator", idx, m_idx))
        img_item.setExpanded(True)


def show_image(index: int):
    if 0 <= index < len(main_window.images):
        main_window.current_image_index = index
        main_window.viewer.load_image(main_window.images[index])
        main_window.viewer.set_markers(main_window.scene[index]["markers"])


def on_tree_selection_changed(current, _previous):
    if not current:
        return
    data = current.data(0, QtCore.Qt.UserRole)
    if not data:
        return
    if data[0] == "image":
        show_image(data[1])
    elif data[0] == "locator":
        show_image(data[1])
        main_window.viewer.set_markers(main_window.scene[data[1]]["markers"])
        # Highlight selected locator
        for i in range(0, len(main_window.viewer.mark_items), 2):
            main_window.viewer.mark_items[i].setBrush(QtGui.QBrush(QtCore.Qt.red))
        idx = data[2]
        if 0 <= idx < len(main_window.viewer.mark_items) // 2:
            ellipse = main_window.viewer.mark_items[idx * 2]
            ellipse.setBrush(QtGui.QBrush(QtCore.Qt.green))


def delete_selected_locator():
    item = main_window.MainTree.currentItem()
    if not item:
        return
    data = item.data(0, QtCore.Qt.UserRole)
    if not data or data[0] != "locator":
        return
    img_idx, loc_idx = data[1], data[2]
    markers = main_window.scene[img_idx]["markers"]
    if 0 <= loc_idx < len(markers):
        markers.pop(loc_idx)
        update_tree()
        show_image(img_idx)


def on_locator_added(x_norm: float, y_norm: float):
    idx = getattr(main_window, "current_image_index", 0)
    if idx >= len(main_window.scene):
        return
    name = f"Loc{main_window.locator_counter}"
    main_window.locator_counter += 1
    main_window.scene[idx]["markers"].append({"name": name, "x": x_norm, "y": y_norm})
    update_tree()
    show_image(idx)



def new_scene():
    main_window.image_paths = []
    main_window.images = []
    main_window.scene = []
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
    main_window.scene = [{"path": p, "markers": []} for p in paths]
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
        "markers": [d["markers"] for d in getattr(main_window, "scene", [])],
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
    loaded_markers = scene.get("markers", [])
    main_window.scene = []
    for i, p in enumerate(main_window.image_paths):
        mlist = loaded_markers[i] if i < len(loaded_markers) else []
        main_window.scene.append({"path": p, "markers": mlist})
    main_window.locator_counter = 1 + sum(len(d["markers"]) for d in main_window.scene)
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
    main_window.viewer.adding_locator = True
    QtWidgets.QMessageBox.information(main_window, "Add Locator", "Click on the image to place a locator.")

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
        main_window.scene = []
        main_window.locator_counter = 1

        # Viewer setup inside MainFrame
        layout = QtWidgets.QVBoxLayout(main_window.MainFrame)
        layout.setContentsMargins(0, 0, 0, 0)
        main_window.viewer = ImageViewer(main_window.MainFrame)
        main_window.viewer.locator_added.connect(on_locator_added)
        layout.addWidget(main_window.viewer)

        # Context action menu for the viewer
        main_window.action_menu = QtWidgets.QMenu(main_window)
        main_window.action_menu.addAction("Add Locator", add_locator)
        main_window.action_menu.addAction("Calibrate", calibrate)
        main_window.action_menu.addAction("Define Worldspace", define_worldspace)
        main_window.action_menu.addAction("Define Reference Distance", define_reference_distance)
        main_window.action_menu.addAction("Add Modeling Locator", add_modeling_locator)

        # Shortcuts for image navigation
        main_window._next_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), main_window)
        main_window._next_shortcut.activated.connect(lambda: show_image(getattr(main_window, "current_image_index", 0) + 1))
        main_window._prev_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), main_window)
        main_window._prev_shortcut.activated.connect(lambda: show_image(getattr(main_window, "current_image_index", 0) - 1))

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
