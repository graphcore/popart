// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/varupdate.hpp"
#include "bindings/op.hpp"
#include "bindings/op/optional.hpp"

#include "bindings/basicoptionals.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/op/varupdate.hpp>
#include <popart/vendored/optional.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

void bindVarupdate(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<VarUpdateOp,
             PyVarUpdateOp<>,
             popart::Op,
             std::shared_ptr<VarUpdateOp>>(sm, "VarUpdateOp")
      .def(py::init<const OperatorIdentifier &, const Op::Settings &>())
      .def("setup", &VarUpdateOp::setup)
      .def("aliases", &VarUpdateOp::aliases)
      .def("modifies", &VarUpdateOp::modifies)
      .def("optimizerInputs", &VarUpdateOp::optimizerInputs)
      .def("isOptimizerOp", &VarUpdateOp::isOptimizerOp)
      .def("growAliasModel", &VarUpdateOp::growAliasModel);

  py::class_<VarUpdateWithUpdaterOp,
             VarUpdateOp,
             PyVarUpdateOp<VarUpdateWithUpdaterOp>,
             std::shared_ptr<VarUpdateWithUpdaterOp>>(sm,
                                                      "VarUpdateWithUpdaterOp")
      .def(py::init<const OperatorIdentifier &, const Op::Settings &>())
      .def("getUpdaterInIndex", VarUpdateWithUpdaterOp::getUpdaterInIndex);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart