// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/optional.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <popart/basicoptionals.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>
#include <popart/vendored/optional.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

void bindOptional(py::module &m) {
  using OptInt = nonstd::optional<int>;

  py::class_<OptInt>(m, "OptionalInt").def(py::init<>()).def(py::init<int>());

  using OptFloat = nonstd::optional<float>;

  py::class_<OptFloat>(m, "OptionalFloat")
      .def(py::init<>())
      .def(py::init<float>());

  py::class_<BasicOptional<popart::DataType, 0>>(m, "OptionalDataType")
      .def(py::init<>())
      .def(py::init<DataType>());

  using OptTensors = nonstd::optional<std::vector<TensorId>>;

  py::class_<OptTensors>(m, "OptionalTensors")
      .def(py::init<>())
      .def(py::init<std::vector<TensorId>>());
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart