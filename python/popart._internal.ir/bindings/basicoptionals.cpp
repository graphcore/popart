// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/basicoptionals.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/basicoptionals.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindBasicOptionals(py::module &m) {
  py::class_<OptionalVGraphId>(m, "OptionalVGraphId")
      .def(py::init<>())
      .def(py::init<int64_t>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalVGraphId::operator bool)
      .def("reset", &OptionalVGraphId::reset);
  py::class_<OptionalPipelineStage>(m, "OptionalPipelineStage")
      .def(py::init<>())
      .def(py::init<int64_t>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalPipelineStage::operator bool)
      .def("reset", &OptionalPipelineStage::reset);
  py::class_<OptionalExecutionPhase>(m, "OptionalExecutionPhase")
      .def(py::init<>())
      .def(py::init<int64_t>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalExecutionPhase::operator bool)
      .def("reset", &OptionalExecutionPhase::reset);
  py::class_<OptionalBatchSerializedPhase>(m, "OptionalBatchSerializedPhase")
      .def(py::init<>())
      .def(py::init<int64_t>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalBatchSerializedPhase::operator bool)
      .def("reset", &OptionalBatchSerializedPhase::reset);
  py::class_<OptionalTensorLocation>(m, "OptionalTensorLocation")
      .def(py::init<>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__bool__", &OptionalTensorLocation::operator bool)
      .def("reset", &OptionalTensorLocation::reset);
}

} // namespace ir
} // namespace _internal
} // namespace popart
