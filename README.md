# MyImageModelerPlugin

This repository contains a prototype image-based camera calibration and export tool for 3ds Max.

## Modules

- `AMmain.py` - Entry point and Qt-based UI integration.
- `AMCameraCalibrate.py` - Legacy chessboard calibration routines.
- `CameraCalibrator.py` - Standalone structure-from-motion module for
  recovering camera poses and 3D points using only 2D tracks. `AMmain.py`
  integrates this module so that pressing the **Calibrate** button uses
  locator marks to estimate camera poses and 3D points.
- `AMUtilities.py` - Helper utilities for loading images and saving scenes.
- `AMUI.ui` - Qt Designer UI file.
- `cpp/Filesystem.cpp` - Minimal C++ implementation of the `.ams` archive format.

## Usage

1. Install the required Python packages:
   ```bash
   pip install PySide2 opencv-python
   ```
2. Build the C++ modules using CMake (requires a compiler and `pybind11` headers):
   ```bash
   mkdir build && cd build
   cmake .. && cmake --build .
   ```
   This produces `CameraCalibrator_cpp` and `Filesystem` Python extensions.
3. Run `AMmain.py` inside 3ds Max's Python environment or as a standalone app for testing.

## Status

This is a starting template and not yet production ready.
