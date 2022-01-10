################## C solver ##############################################
add_library(_c_solver_lib STATIC _c_solver_lib.c)
target_include_directories(_c_solver_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
python_extension_module(_c_solver_lib)


################## C++ solver ############################################
pybind11_add_module(_cpp_solver_lib _cpp_solver_lib.cpp)
install(TARGETS _cpp_solver_lib LIBRARY DESTINATION cofi/cofi_solvers/linear_regression)


################## Fortran 77 solver #####################################
# set(f2py_module_name "_cylinder_methods")
# set(fortran_src_file "${CMAKE_CURRENT_SOURCE_DIR}/cylinder_methods.f90")
# set(generated_module_file ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name}${PYTHON_EXTENSION_MODULE_SUFFIX})

set(f2py_module_name "_f77_solver_lib")
set(fortran_src_file "${CMAKE_CURRENT_SOURCE_DIR}/_f77_solver_lib.f")
set(generated_module_file ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name}${PYTHON_EXTENSION_MODULE_SUFFIX})
add_custom_target(${f2py_module_name} ALL DEPENDS ${generated_module_file})
add_custom_command(
  OUTPUT ${generated_module_file}
  COMMAND ${F2PY_EXECUTABLE}
    -m ${f2py_module_name}
    -c
    ${fortran_src_file}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
install(FILES ${generated_module_file} DESTINATION cofi/cofi_solvers/linear_regression)
