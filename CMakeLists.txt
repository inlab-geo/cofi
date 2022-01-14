################################################################################
# Project configuration and settings
################################################################################
cmake_minimum_required(VERSION 3.11.0)

project(CoFI)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(F2PY REQUIRED)
# find_package(pybind11 REQUIRED)
# find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# PyBind11 related
include(GNUInstallDirs)
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.6.2.tar.gz
  URL_HASH SHA256=8ff2fff22df038f5cd02cea8af56622bc67f5b64534f1b83b9f133b8366acff2
)
FetchContent_MakeAvailable(pybind11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


################################################################################
# Libraries
################################################################################
add_subdirectory(cofi/cofi_solvers/linear_regression)
