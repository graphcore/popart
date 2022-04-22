// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/transforms/transform.hpp"

#include <array>
#include <initializer_list>
#include <pybind11/pybind11.h>
#include <popart/transforms/transform.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace transforms {

void bindTransform(py::module &m) {

  py::class_<Transform, PyTransform<Transform>>(m, "Transform")
      .def(py::init<>())
      .def("apply", &Transform::apply)
      .def("getId", &Transform::getId)
      .def("getName", &Transform::getName)
      .def("applyTransform", &Transform::applyTransform)
      .def("registerTransform", &Transform::registerTransform);
}

} // namespace transforms
} // namespace ir
} // namespace _internal
} // namespace popart
