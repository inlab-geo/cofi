################################################################################
# Project configuration and settings
################################################################################
message(STATUS "    reach root level")
cmake_minimum_required(VERSION 3.11.0)

project(CoFI)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


################################################################################
# Libraries
################################################################################
add_subdirectory(cofi/cofi_solvers/linear_regression)

message(STATUS "    finished root level")
