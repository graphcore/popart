// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/opidentifier.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/opidentifier.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindOpIdentifier(py::module &m) {
  py::class_<NumInputs>(m, "NumInputs")
      .def(py::init<>())
      .def(py::init<int>())
      .def(py::init<int, int>())
      .def_readwrite("min", &NumInputs::min)
      .def_readwrite("max", &NumInputs::min);
  py::class_<OperatorIdentifier>(m, "OperatorIdentifier", py::module_local())
      .def(py::init<const OpDomain, const OpType, OpVersion, NumInputs, int>(),
           py::arg("domain"),
           py::arg("type"),
           py::arg("version"),
           py::arg("inputs")  = NumInputs(),
           py::arg("outputs") = 0)
      .def_readonly("domain", &OperatorIdentifier::domain)
      .def_readonly("type", &OperatorIdentifier::type)
      .def_readonly("version", &OperatorIdentifier::version)
      .def_readonly("numInputs", &OperatorIdentifier::numInputs)
      .def_readonly("numOutputs", &OperatorIdentifier::numOutputs)
      .def(py::self < py::self)
      .def(py::self == py::self)
      .def(py::self != py::self);
}

} // namespace ir
} // namespace _internal
} // namespace popart
