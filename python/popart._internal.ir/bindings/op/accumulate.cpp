// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/accumulate.hpp"

#include <array>
#include <initializer_list>
#include <memory>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/op/accumulate.hpp>

#include "bindings/op/varupdate.hpp"
#include "popart/alias/aliasmodel.hpp" // IWYU pragma: keep
#include "popart/op.hpp"

namespace py = pybind11;

namespace popart {
class OptimizerValue;
class VarUpdateWithUpdaterOp;
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {

void bindAccumulate(py::module &m) {

  py::enum_<AccumulationType>(m, "AccumulationType", py::module_local())
      .value("Add", AccumulationType::Add)
      .value("DampenedAdd", AccumulationType::DampenedAdd)
      .value("DampenedAddSquare", AccumulationType::DampenedAddSquare)
      .value("DecayAdd", AccumulationType::DecayAdd)
      .value("DecayAddSquare", AccumulationType::DecayAddSquare)
      .value("MovingAverage", AccumulationType::MovingAverage)
      .value("MovingAverageSquare", AccumulationType::MovingAverageSquare)
      .value("Infinity", AccumulationType::Infinity)
      .value("Mean", AccumulationType::Mean);

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<AccumulateBaseOp,
             VarUpdateWithUpdaterOp,
             PyVarUpdateOp<AccumulateBaseOp>,
             std::shared_ptr<AccumulateBaseOp>>(sm, "AccumulateBaseOp")
      .def(py::init<const OperatorIdentifier &,
                    AccumulationType,
                    OptimizerValue,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("accumulationType"),
           py::arg("optimizer_value"),
           py::arg("settings"))
      .def("getAccumulationType", &AccumulateBaseOp::getAccumulationType);

  py::class_<AccumulateOp,
             AccumulateBaseOp,
             PyVarUpdateOp<AccumulateOp>,
             std::shared_ptr<AccumulateOp>>(sm, "AccumulateOp")
      .def(py::init<AccumulationType, OptimizerValue, const Op::Settings &>(),
           py::arg("accumulationType"),
           py::arg("factor"),
           py::arg("settings"));

  py::class_<SparseAccumulateOp,
             AccumulateBaseOp,
             PyVarUpdateOp<SparseAccumulateOp>,
             std::shared_ptr<SparseAccumulateOp>>(sm, "SparseAccumulateOp")
      .def(py::init<AccumulationType,
                    const OptimizerValue &,
                    unsigned,
                    const Op::Settings &>(),
           py::arg("accumulationType"),
           py::arg("optimizer_value"),
           py::arg("axis"),
           py::arg("settings"))
      .def("getAxis", &SparseAccumulateOp::getAxis)
      .def("getFactor", &SparseAccumulateOp::getFactor);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
