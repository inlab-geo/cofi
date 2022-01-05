add_subdirectory(lib)

add_cython_target(_c_solver _c_solver.pyx)
add_library(_c_solver MODULE ${_c_solver})
python_extension_module(_c_solver)
target_link_libraries(_c_solver _c_solver_lib)

include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(_c_solver ${PYTHON_LIBRARIES} Python3::NumPy)
install(TARGETS _c_solver LIBRARY DESTINATION cofi/cofi_solvers/linear_regression)
