cmake_minimum_required(VERSION 3.14)
project(Filesystem)

set(CMAKE_CXX_STANDARD 17)

# Указание Python вручную (только 3.9!)
set(PYBIND11_FINDPYTHON OLD)
set(PYTHON_EXECUTABLE "C:/Users/kholk/AppData/Local/Programs/Python/Python39/python.exe")

# Встроенный pybind11
add_subdirectory(pybind11)

# Инклуды
include_directories(${CMAKE_SOURCE_DIR}/cpp)

# miniz.c — обычный C-шный файл
set_source_files_properties(cpp/miniz.c PROPERTIES LANGUAGE C)

# Модуль
pybind11_add_module(Filesystem
    cpp/Filesystem.cpp
    cpp/md5.cpp
    cpp/miniz.c
)
