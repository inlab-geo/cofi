#include <iostream>
#include <pybind11/pybind11.h>


// ----------------
// Regular C++ code
// ----------------

void hello() {
    std::cout << "Hello, world!" << std::endl;
}


// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(_cpp_solver_lib, m) {
    m.doc() = "_cpp_solver_lib";
    m.def("hello", &hello, "Prints \"Hello, world!\"");
}
