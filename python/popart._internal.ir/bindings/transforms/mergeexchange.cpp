// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
#include <cstddef>
#include <initializer_list>
#include <pybind11/cast.h>       // IWYU pragma: keep
#include <pybind11/functional.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <string>
#include <vector>
#include <popart/transforms/mergeexchange.hpp>

#include "bindings/transforms/transform.hpp"
#include "popart/graph.hpp" // IWYU pragma: keep

namespace py = pybind11;

namespace popart {
class Transform;

namespace _internal {
namespace ir {
namespace transforms {
void bindMergeExchange(py::module &m) {

  py::class_<MergeExchange, Transform, PyTransform<MergeExchange>>(
      m, "MergeExchange")
      .def(py::init<>())
      .def("id", &MergeExchange::id)
      .def("apply", &MergeExchange::apply)
      .def("applyToOps",
           &MergeExchange::applyToOps,
           py::return_value_policy::reference)
      .def("getId", &MergeExchange::getId)
      .def("getName", &MergeExchange::getName);
}

} // namespace transforms
} // namespace ir
} // namespace _internal
} // namespace popart
