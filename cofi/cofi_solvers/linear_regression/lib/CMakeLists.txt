add_library(_c_solver_lib STATIC _c_solver_lib.c)
target_include_directories(_c_solver_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
python_extension_module(_c_solver_lib)
