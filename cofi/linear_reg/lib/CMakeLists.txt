################## C solver ##############################################
add_library(_c_solver_lib STATIC _c_solver_lib.c)
target_include_directories(_c_solver_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
python_extension_module(_c_solver_lib)


################## C++ solver ############################################
pybind11_add_module(_cpp_solver_lib _cpp_solver_lib.cpp)
install(TARGETS _cpp_solver_lib LIBRARY DESTINATION cofi/linear_reg)


################## Fortran 77 solver #####################################
set(f2py_module_name_77 "_f77_solver_lib")
set(fortran_77_src_file "${CMAKE_CURRENT_SOURCE_DIR}/_f77_solver_lib.f")
set(generated_module_file_77 ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name_77}${PYTHON_EXTENSION_MODULE_SUFFIX})
add_custom_target(${f2py_module_name_77} ALL DEPENDS ${generated_module_file_77})
add_custom_command(
  OUTPUT ${generated_module_file_77}
  COMMAND ${F2PY_EXECUTABLE}
    -m ${f2py_module_name_77}
    -c
    ${fortran_77_src_file}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
install(FILES ${generated_module_file_77} DESTINATION cofi/linear_reg)


################## Fortran 90 solver #####################################
set(f2py_module_name_90 "_f90_solver_lib")
set(fortran_90_src_file "${CMAKE_CURRENT_SOURCE_DIR}/_f90_solver_lib.f90")
set(generated_module_file_90 ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name_90}${PYTHON_EXTENSION_MODULE_SUFFIX})
add_custom_target(${f2py_module_name_90} ALL DEPENDS ${generated_module_file_90})
add_custom_command(
  OUTPUT ${generated_module_file_90}
  COMMAND ${F2PY_EXECUTABLE}
    -m ${f2py_module_name_90}
    -c
    ${fortran_90_src_file}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
install(FILES ${generated_module_file_90} DESTINATION cofi/linear_reg)
