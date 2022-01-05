################################################################################
# Project configuration and settings
################################################################################
cmake_minimum_required(VERSION 3.11.0)

project(CoFI)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

# if(SKBUILD)
#   set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
#   set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
#   set(Python_LIBRARY "${PYTHON_LIBRARY}")
# endif()

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# find_package(PythonLibs REQUIRED)
# find_package(NumPy REQUIRED)
# find_package(Python COMPONENTS NumPy)
# message(STATUS "Python_NumPy_FOUND: ${Python_NumPy_FOUND}")
# message(STATUS "Python_NumPy_INCLUDE_DIRS: ${Python_NumPy_INCLUDE_DIRS}")
# message(STATUS "Python_NumPy_VERSION: ${Python_NumPy_VERSION}")


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


################################################################################
# Libraries
################################################################################
add_subdirectory(cofi/cofi_solvers/linear_regression)
