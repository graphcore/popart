// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/accumulate.hpp"
#include "bindings/op.hpp"
#include "bindings/op/optional.hpp"

#include "bindings/basicoptionals.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/op/accumulate.hpp>
#include <popart/vendored/optional.hpp>

namespace py = pybind11;

namespace popart {
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

  py::class_<AccumulateOp, popart::Op, std::shared_ptr<AccumulateOp>>(
      sm, "AccumulateOp")
      .def(py::init<AccumulationType, OptimizerValue, const Op::Settings &>(),
           py::arg("accumulationType"),
           py::arg("factor"),
           py::arg("settings"))
      .def("getAccumulationType", &AccumulateBaseOp::getAccumulationType);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart