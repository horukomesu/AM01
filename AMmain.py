"""Main entry point for MyImageModelerPlugin.

This script loads the Qt UI designed in ``AMUI.ui`` and wires user
interactions to the calibration and utility modules.  It is intended to
run inside Autodesk 3ds Max 2023+ but can also be executed as a
standalone Python application for testing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from PySide2 import QtWidgets, QtCore, QtGui, QtUiTools

# Ensure the package directory is in sys.path so modules import correctly
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
        loader.setWorkingDirectory(str(ui_path.parent))
        ui_file = QtCore.QFile(str(ui_path))
        ui_file.open(QtCore.QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()
        self.setCentralWidget(self.ui)

        # Connect toolbar buttons
        self.ui.btnAddLoc.clicked.connect(self.add_locator)
        self.ui.btnCalibrate.clicked.connect(self.calibrate)
        self.ui.btnDFWS.clicked.connect(self.define_worldspace)
        self.ui.btnDFMM.clicked.connect(self.define_reference_distance)
        self.ui.btnLocMod.clicked.connect(self.add_modeling_locator)

        # Connect menu actions
        self.ui.actionNEW.triggered.connect(self.new_scene)
        self.ui.actionOpen.triggered.connect(self.load_scene)
        self.ui.actionSave.triggered.connect(self.save_scene)
        self.ui.actionSave_As.triggered.connect(self.save_scene_as)
        self.ui.actionLoad.triggered.connect(self.import_images)
        self.ui.actionRecent_Projects.triggered.connect(self.open_recent_project)
        self.ui.actionPreferences.triggered.connect(self.preferences)
        self.ui.actionUndo.triggered.connect(self.undo)
        self.ui.actionRedo.triggered.connect(self.redo)

        self.image_paths: List[str] = []
        self.images: List[QtGui.QImage] = []

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    def new_scene(self) -> None:
        """Clear the current scene."""
        self.image_paths.clear()
        self.images.clear()
        self.ui.MainTree.clear()
        QtWidgets.QMessageBox.information(self, "New", "Started a new scene.")

    def import_images(self) -> None:
        """Import images and populate the Scene Browser."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.tif)"
        )
        paths = AMUtilities.verify_paths(paths)
        self.image_paths = paths
        self.images = AMUtilities.load_images(paths)
        self.ui.MainTree.clear()
        for p in self.image_paths:
            QtWidgets.QTreeWidgetItem(self.ui.MainTree, [Path(p).name])

    def save_scene(self) -> None:
        """Save current scene to JSON."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Scene", "", "JSON (*.json)"
        )
        if not path:
            return
        scene = {"images": self.image_paths}
        AMUtilities.save_scene(scene, path)

    def save_scene_as(self) -> None:
        """Save scene using Save As."""
        self.save_scene()

    def load_scene(self) -> None:
        """Load scene from JSON."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Scene", "", "JSON (*.json)"
        )
        if not path:
            return
        scene = AMUtilities.load_scene(path)
        self.image_paths = scene.get("images", [])
        self.images = AMUtilities.load_images(self.image_paths)
        self.ui.MainTree.clear()
        for p in self.image_paths:
            QtWidgets.QTreeWidgetItem(self.ui.MainTree, [Path(p).name])

    def open_recent_project(self) -> None:
        QtWidgets.QMessageBox.information(self, "Recent", "Recent projects not implemented.")

    # ------------------------------------------------------------------
    # Edit operations
    # ------------------------------------------------------------------
    def preferences(self) -> None:
        QtWidgets.QMessageBox.information(self, "Preferences", "Preferences dialog not implemented.")

    def undo(self) -> None:
        QtWidgets.QMessageBox.information(self, "Undo", "Undo not implemented.")

    def redo(self) -> None:
        QtWidgets.QMessageBox.information(self, "Redo", "Redo not implemented.")

    # ------------------------------------------------------------------
    # Toolbar / workflow actions
    # ------------------------------------------------------------------
    def add_locator(self) -> None:
        QtWidgets.QMessageBox.information(self, "Add Locator", "Create/Move marker not implemented.")

    def calibrate(self) -> None:
        QtWidgets.QMessageBox.information(self, "Calibrate", "Calibration routine not implemented.")

    def define_worldspace(self) -> None:
        QtWidgets.QMessageBox.information(self, "Worldspace", "Define worldspace not implemented.")

    def define_reference_distance(self) -> None:
        QtWidgets.QMessageBox.information(self, "Reference Distance", "Define reference distance not implemented.")

    def add_modeling_locator(self) -> None:
        QtWidgets.QMessageBox.information(self, "Modeling Locator", "Add modeling locator not implemented.")


def run() -> None:
    """Entry point to launch the application."""
    app = QtWidgets.QApplication.instance()
    own_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        own_app = True

    ui_path = Path(__file__).with_name("AMUI.ui")
    window = MainWindow(ui_path)
    window.show()

    if own_app:
        app.exec_()


if __name__ == "__main__":
    run()
