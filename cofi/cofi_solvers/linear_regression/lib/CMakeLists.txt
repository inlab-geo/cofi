add_library(_c_solver_lib STATIC _c_solver_lib.c)
target_include_directories(_c_solver_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
python_extension_module(_c_solver_lib)

pybind11_add_module(_cpp_solver_lib _cpp_solver_lib.cpp)
# add_library(_cpp_solver_lib MODULE _cpp_solver_lib.cpp)
# python_extension_module(_cpp_solver_lib)
install(TARGETS _cpp_solver_lib LIBRARY DESTINATION cofi/cofi_solvers/linear_regression)
