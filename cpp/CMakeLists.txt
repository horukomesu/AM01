cmake_minimum_required(VERSION 3.14)
project(CameraCalibratorCPP)

find_package(pybind11 REQUIRED)
find_package(colmap REQUIRED)

pybind11_add_module(CameraCalibrator_cpp CameraCalibrator.cpp)

target_link_libraries(CameraCalibrator_cpp PRIVATE colmap)
