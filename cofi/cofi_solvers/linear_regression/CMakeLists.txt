add_library(_c_solver_lib MODULE lib/_c_solver.c)
python_extension_module(_c_solver_lib)
install(TARGETS _c_solver_lib LIBRARY DESTINATION cofi/cofi_solvers/linear_regression/lib)

add_cython_target(_c_solver CXX)
add_library(_c_solver MODULE ${_c_solver})
python_extension_module(_c_solver)

install(TARGETS _c_solver LIBRARY DESTINATION cofi/cofi_solvers/linear_regression)

