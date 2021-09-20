// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/optional.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <popart/basicoptionals.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/vendored/optional.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

void bindOptional(py::module &m) {
  using OptInt = nonstd::optional<int>;

  py::class_<OptInt>(m, "OptionalInt")
      // .def(py::init<>()) <- Don't bind, leads to bad optional access.
      .def(py::init<int>())
      .def("__str__", [](OptInt &self) { return std::to_string(self.value()); })
      .def("__repr__",
           [](OptInt &self) { return std::to_string(self.value()); });

  using OptFloat = nonstd::optional<float>;

  py::class_<OptFloat>(m, "OptionalFloat")
      .def(py::init<>())
      .def(py::init<float>())
      .def("__str__",
           [](OptFloat &self) { return std::to_string(self.value()); })
      .def("__repr__",
           [](OptFloat &self) { return std::to_string(self.value()); });

  py::class_<BasicOptional<popart::DataType, 0>>(m, "OptionalDataType")
      .def(py::init<>())
      .def(py::init<DataType>());
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart