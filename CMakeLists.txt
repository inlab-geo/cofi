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
# include(GNUInstallDirs)
# include(FetchContent)
# FetchContent_Declare(
#   pybind11
#   URL https://github.com/pybind/pybind11/archive/refs/tags/v2.6.2.tar.gz
#   URL_HASH SHA256=8ff2fff22df038f5cd02cea8af56622bc67f5b64534f1b83b9f133b8366acff2
# )
# FetchContent_MakeAvailable(pybind11)

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


################################################################################
# Solver Libraries
################################################################################
add_subdirectory(cofi/linear_reg)


################################################################################
# Objective Libraries
################################################################################
add_subdirectory(cofi/cofi_objective/examples/lib_rfc)
