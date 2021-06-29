// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/graphid.hpp"

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
      .def(py::self < py::self);
}

} // namespace ir
} // namespace _internal
} // namespace popart
