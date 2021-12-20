cmake_minimum_required(VERSION 3.11.0)

project(cofi)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

add_subdirectory(cofi/cofi_solvers/linear_regression)
