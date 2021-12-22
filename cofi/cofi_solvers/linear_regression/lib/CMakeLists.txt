message(STATUS "            reach cofi/cofi_solvers/linear_regression/lib/")

add_library(_c_solver_lib STATIC _c_solver_lib.c)
target_include_directories(_c_solver_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS _c_solver_lib LIBRARY DESTINATION cofi/cofi_solvers/linear_regression/lib)

message(STATUS "            finished cofi/cofi_solvers/linear_regression/lib/")
