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

def new_scene():
    main_window.image_paths = []
    main_window.images = []
    main_window.MainTree.clear()
    QtWidgets.QMessageBox.information(main_window, "New", "Started a new scene.")

def import_images():
    paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
        main_window, "Select Images", "", "Images (*.png *.jpg *.jpeg *.tif)"
    )
    paths = AMUtilities.verify_paths(paths)
    main_window.image_paths = paths
    main_window.images = AMUtilities.load_images(paths)
    main_window.MainTree.clear()
    for p in main_window.image_paths:
        QtWidgets.QTreeWidgetItem(main_window.MainTree, [Path(p).name])

def save_scene():
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        main_window, "Save Scene", "", "JSON (*.json)"
    )
    if not path:
        return
    scene = {"images": getattr(main_window, "image_paths", [])}
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
    main_window.MainTree.clear()
    for p in main_window.image_paths:
        QtWidgets.QTreeWidgetItem(main_window.MainTree, [Path(p).name])

def open_recent_project():
    QtWidgets.QMessageBox.information(main_window, "Recent", "Recent projects not implemented.")

def preferences():
    QtWidgets.QMessageBox.information(main_window, "Preferences", "Preferences dialog not implemented.")

def undo():
    QtWidgets.QMessageBox.information(main_window, "Undo", "Undo not implemented.")

def redo():
    QtWidgets.QMessageBox.information(main_window, "Redo", "Redo not implemented.")

def add_locator():
    QtWidgets.QMessageBox.information(main_window, "Add Locator", "Create/Move marker not implemented.")

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
