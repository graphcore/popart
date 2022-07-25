// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/transforms/decomposesum.hpp"
#include "bindings/transforms/transform.hpp"

#include <array>
#include <initializer_list>
#include <pybind11/pybind11.h>
#include <popart/transforms/decomposesum.hpp>

#include <popart/graph.hpp> // IWYU pragma: keep

namespace py = pybind11;

namespace popart {
class Transform;

namespace _internal {
namespace ir {
namespace transforms {
void bindDecomposeSum(py::module &m) {

  py::class_<DecomposeSum, Transform, PyTransform<DecomposeSum>>(m,
                                                                 "DecomposeSum")
      .def(py::init<>())
      .def("apply", &DecomposeSum::apply)
      .def("getId", &DecomposeSum::getId)
      .def("getName", &DecomposeSum::getName);
}

} // namespace transforms
} // namespace ir
} // namespace _internal
} // namespace popart
