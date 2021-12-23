message(STATUS "        reach cofi/cofi_solvers/linear_regression/")
add_subdirectory(lib)
message(STATUS "        back to cofi/cofi_solvers/linear_regression/")

# add_cython_target(_c_solver.pyx)
# add_library(_c_solver MODULE ${_c_solver})
# target_link_libraries(_c_solver PUBLIC _c_solver_lib)
# install(TARGETS _c_solver LIBRARY DESTINATION cofi/cofi_solvers/linear_regression)

add_cython_target(_c_solver _c_solver.pyx)
add_library(_c_solver MODULE ${_c_solver})
python_extension_module(_c_solver)
target_link_libraries(_c_solver _c_solver_lib)
install(TARGETS _c_solver LIBRARY DESTINATION cofi/cofi_solvers/linear_regression)

message(STATUS "        finished cofi/cofi_solvers/linear_regression")
