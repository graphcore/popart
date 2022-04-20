// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/graphid.hpp"

#include <initializer_list>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <string>
#include <popart/graphid.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindGraphId(py::module &m) {
  py::class_<GraphId>(m, "GraphId")
      .def(py::init<const std::string>())
      .def(py::self < py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("str", &GraphId::str, py::return_value_policy::reference)
      .def("__str__", &GraphId::str, py::return_value_policy::reference)
      .def("__hash__", [](GraphId &self) {
        std::hash<std::string> hasher;
        auto hashed = hasher(self.str());
        return hashed;
      });
  // Allow for string to be implicitly converted to a GraphId.
  py::implicitly_convertible<std::string, GraphId>();
}

} // namespace ir
} // namespace _internal
} // namespace popart
