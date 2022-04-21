// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/transforms/prune.hpp"
#include "bindings/transforms/transform.hpp"

#include <pybind11/pybind11.h>
#include <popart/graph.hpp>
#include <popart/transforms/prune.hpp>
#include <popart/transforms/transform.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace transforms {
void bindPrune(py::module &m) {

  py::class_<Prune, Transform, PyTransform<Prune>>(m, "Prune")
      .def(py::init<>())
      .def("id", &Prune::id)
      .def("apply", [](Prune &self, Graph &graph) { return self.apply(graph); })
      .def("getId", &Prune::getId)
      .def("getName", &Prune::getName);
}

} // namespace transforms
} // namespace ir
} // namespace _internal
} // namespace popart
