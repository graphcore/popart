#include <pybind11/pybind11.h>

#include <popart/ir.hpp>

// Shorthand for namespace.
namespace py = pybind11;

PYBIND11_MODULE(popart_internal_ir, m) {
  m.doc() = "This module is an internal PopART API (`popart._internal.ir`) "
            "that is used to implement the public `popart.ir` API. This "
            "internal API is not intended for public use and may change "
            "between releases with no guarantee of backwards compatibility "
            "or deprecation periods.";

  py::class_<popart::Ir> cls(m, "Ir");
  cls.def(py::init<>());
}