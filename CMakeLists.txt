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
# find_package(Python3 COMPONENTS Interpreter REQUIRED)

# PyBind11 related
# include(GNUInstallDirs)
# include(FetchContent)
# FetchContent_Declare(
#   pybind11
#   URL https://github.com/pybind/pybind11/archive/refs/tags/v2.9.1.tar.gz
#   URL_HASH SHA256=c6160321dc98e6e1184cc791fbeadd2907bb4a0ce0e447f2ea4ff8ab56550913
# )
# FetchContent_MakeAvailable(pybind11)

# Add proper -fPIC and MINGW handling
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (NOT MINGW)
        set(common "-fPIC")
    else()
        set(common "-D_hypot=hypot -Doff64_t=_off64_t")
        if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
            set(common "${common} -DMS_WIN64")
        endif()
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${common}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${common}")
endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(common "-fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${common}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${common}")
endif ()


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
