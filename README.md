# MyImageModelerPlugin

This repository contains a prototype image-based camera calibration and export tool for 3ds Max.

## Modules

- `AMmain.py` - Entry point and Qt-based UI integration.
- `AMCameraCalibrate.py` - Core calibration logic built on OpenCV.
- `AMUtilities.py` - Helper utilities for loading images and saving scenes.
- `AMUI.ui` - Qt Designer UI file.

## Usage

1. Install the required Python packages:
   ```bash
   pip install PySide2 opencv-python
   ```
2. Run `AMmain.py` inside 3ds Max's Python environment or as a standalone app for testing.

## Status

This is a starting template and not yet production ready.
