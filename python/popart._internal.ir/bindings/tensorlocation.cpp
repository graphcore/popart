// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/tensorlocation.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/tensorlocation.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindTensorLocation(py::module &m) {
  py::enum_<TileSet>(m, "TileSet", py::module_local())
      .value("Compute", TileSet::Compute)
      .value("IO", TileSet::IO)
      .value("Undefined", TileSet::Undefined)
      .value("N", TileSet::N);
  py::class_<TensorLocation>(m, "TensorLocation", py::module_local())
      .def(py::init<>())
      .def("operator=", &TensorLocation::operator=)
      .def("operator==", &TensorLocation::operator==)
      .def("operator!=", &TensorLocation::operator!=)
      .def("serialize", &TensorLocation::serialize)
      .def("isRemote", &TensorLocation::isRemote);
}

} // namespace ir
} // namespace _internal
} // namespace popart
