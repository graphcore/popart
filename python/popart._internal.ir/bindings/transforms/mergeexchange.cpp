// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/transforms/autodiff.hpp"
#include "bindings/transforms/transform.hpp"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/graph.hpp>
#include <popart/transforms/mergeexchange.hpp>

#include <popart/transforms/transform.hpp>

namespace py = pybind11;

namespace popart {
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
