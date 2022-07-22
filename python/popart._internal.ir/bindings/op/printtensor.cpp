// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/printtensor.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/op/printtensor.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensordebuginfo.hpp"

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindPrintTensor(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<PrintTensorOp, popart::Op, std::shared_ptr<PrintTensorOp>>(
      sm, "PrintTensorOp")
      .def(py::init<const popart::OperatorIdentifier &,
                    bool,
                    bool,
                    const std::string &,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("printSelf"),
           py::arg("printGradient"),
           py::arg("title"),
           py::arg("settings"))
      .def("getTitle", &PrintTensorOp::getTitle)
      .def("setTitle", &PrintTensorOp::setTitle);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
