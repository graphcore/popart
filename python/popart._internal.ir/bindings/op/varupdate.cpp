// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
#include <initializer_list>
#include <memory>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <vector>
#include <popart/op/varupdate.hpp>

#include "bindings/op/varupdate.hpp"
#include "popart/alias/aliasmodel.hpp" // IWYU pragma: keep
#include "popart/op.hpp"
#include "popart/region.hpp" // IWYU pragma: keep

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
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
