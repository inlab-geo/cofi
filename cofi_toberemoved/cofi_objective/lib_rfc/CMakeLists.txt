################## Compiling Fortran rfc library #####################################
set(f2py_module_name_rfc "_rfc")
set(fortran_src_file_rfc "${CMAKE_CURRENT_SOURCE_DIR}/RF.F90")
file(GLOB_RECURSE fortran_src_file_rfcsubs "${CMAKE_CURRENT_SOURCE_DIR}/RFsubs/*.f*")
set(generated_module_file_rfc ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name_rfc}${PYTHON_EXTENSION_MODULE_SUFFIX})
add_custom_target(${f2py_module_name_rfc} ALL DEPENDS ${generated_module_file_rfc})
add_custom_command(
  OUTPUT ${generated_module_file_rfc}
  COMMAND ${F2PY_EXECUTABLE}
    -m ${f2py_module_name_rfc}
    -c
    ${fortran_src_file_rfc} 
    ${fortran_src_file_rfcsubs}
    --opt='-O3'
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
install(FILES ${generated_module_file_rfc} DESTINATION cofi/cofi_objective)
