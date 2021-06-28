// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/graph.hpp"

#include <pybind11/pybind11.h>
#include <popart/graph.hpp>
#include <popart/graphid.hpp>
#include <popart/ir.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindGraph(py::module &m) {
  py::class_<Graph>(m, "Graph")
      .def(py::init<Ir &, const GraphId &>())
      .def_readonly("id", &Graph::id);
}

} // namespace ir
} // namespace _internal
} // namespace popart
