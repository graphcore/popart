// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <initializer_list>
#include <pybind11/cast.h>
#include <pybind11/numpy.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <utility>
#include <popart/optimizervalue.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

void bindOptimizerValue(py::module &m) {
  py::class_<OptimizerValue> optimizerValue(m, "OptimizerValue");

  optimizerValue
      .def(py::init<float, bool>(), py::arg("val"), py::arg("isConst"))
      .def(py::init<float>(), py::arg("val"))
      .def(py::init<>())
      .def(py::init<std::pair<float, bool>>())
      .def("val", &OptimizerValue::val)
      .def("isConst", &OptimizerValue::isConst);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
