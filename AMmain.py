"""Main entry point for MyImageModelerPlugin.

This script loads the Qt UI designed in ``AMUI.ui`` and wires user
interactions to the calibration and utility modules.

The plugin is intended to run inside Autodesk 3ds Max 2023+ but can be
executed as a standalone Python application for testing the UI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from PySide2 import QtWidgets, QtCore, QtGui, QtUiTools

# Ensure this directory is on sys.path so local modules can be imported
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import AMUtilities
import AMCameraCalibrate


class MainWindow(QtWidgets.QMainWindow):
    """Main application window created from the Qt Designer UI."""

    def __init__(self, ui_path: Path) -> None:
        super().__init__()
        loader = QtUiTools.QUiLoader()
        ui_file = QtCore.QFile(str(ui_path))
        ui_file.open(QtCore.QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()
        self.setCentralWidget(self.ui)

        self.ui.importImagesButton.clicked.connect(self.import_images)
        self.ui.saveSceneButton.clicked.connect(self.save_scene)
        self.ui.loadSceneButton.clicked.connect(self.load_scene)
        self.ui.calibrateButton.clicked.connect(self.calibrate)

        self.images: List[QtGui.QImage] = []
        self.image_paths: List[str] = []

    def import_images(self) -> None:
        """Import images and show thumbnails."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.tif)"
        )
        paths = AMUtilities.verify_paths(paths)
        self.image_paths = paths
        self.images = AMUtilities.load_images(paths)
        for img in self.images:
            item = QtWidgets.QListWidgetItem()
            icon = QtGui.QIcon(QtGui.QPixmap.fromImage(img))
            item.setIcon(icon)
            self.ui.imageList.addItem(item)

    def save_scene(self) -> None:
        """Save current scene to JSON."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Scene", "", "JSON (*.json)")
        if not path:
            return
        scene = {
            "images": self.image_paths,
            # TODO: collect locators and camera params
        }
        AMUtilities.save_scene(scene, path)

    def load_scene(self) -> None:
        """Load scene from JSON."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Scene", "", "JSON (*.json)")
        if not path:
            return
        scene = AMUtilities.load_scene(path)
        self.image_paths = scene.get("images", [])
        self.ui.imageList.clear()
        self.images = AMUtilities.load_images(self.image_paths)
        for img in self.images:
            item = QtWidgets.QListWidgetItem()
            icon = QtGui.QIcon(QtGui.QPixmap.fromImage(img))
            item.setIcon(icon)
            self.ui.imageList.addItem(item)
        # TODO: load locators and camera params

    def calibrate(self) -> None:
        """Run camera calibration using current locator data."""
        # Placeholder: actual collection of points not yet implemented
        QtWidgets.QMessageBox.information(self, "Calibrate", "Calibration routine not implemented.")


def run() -> None:
    """Entry point to launch the application."""
    app = QtWidgets.QApplication(sys.argv)
    ui_path = Path(__file__).with_name("AMUI.ui")
    window = MainWindow(ui_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
