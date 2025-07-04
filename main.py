from PyQt6.QtWidgets import QApplication
from ui.ui_mainwindow import Ui_MainWindow
from core.viewer import ImageViewer
from core.tree_logic import setup_tree_events
from core.calibration import setup_calibration
from core.settings import load_settings, save_settings
from utils import am_utilities
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout
import sys

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # custom state
        self.image_paths = []
        self.images = []
        self.locators = []
        self.selected_locator = None
        self.locator_mode = False
        self.image_errors = {}

        # viewer
        layout = QVBoxLayout(self.MainFrame)
        layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = ImageViewer(self.MainFrame)
        layout.addWidget(self.viewer)

        # дерево и действия
        setup_tree_events(self)
        setup_calibration(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 900)
    win.show()
    sys.exit(app.exec())
