// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/enums.hpp"

#include <initializer_list>
#include <pybind11/attr.h>
#include <pybind11/numpy.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/adam.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/init.hpp>

#include "popart/op.hpp"

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

void bindEnums(py::module &m) {
  py::enum_<ReductionType>(m, "ReductionType", py::module_local())
      .value("Sum", ReductionType::Sum)
      .value("Mean", ReductionType::Mean)
      .value("NoReduction", ReductionType::NoReduction)
      .value("N", ReductionType::N);
  py::enum_<InitType>(m, "InitType", py::module_local())
      .value("NoInit", InitType::NoInit)
      .value("Zero", InitType::Zero);
  py::enum_<CollectiveOperator>(m, "CollectiveOperator", py::module_local())
      .value("Add", CollectiveOperator::Add)
      .value("Mean", CollectiveOperator::Mean)
      .value("Mul", CollectiveOperator::Mul)
      .value("Min", CollectiveOperator::Min)
      .value("Max", CollectiveOperator::Max)
      .value("LogicalAnd", CollectiveOperator::LogicalAnd)
      .value("LogicalOr", CollectiveOperator::LogicalOr)
      .value("SquareAdd", CollectiveOperator::SquareAdd)
      .value("Local", CollectiveOperator::Local);
  py::enum_<AdamMode>(m, "AdamMode", py::module_local())
      .value("Adam", AdamMode::Adam)
      .value("AdamNoBias", AdamMode::AdamNoBias)
      .value("AdaMax", AdamMode::AdaMax)
      .value("Lamb", AdamMode::Lamb)
      .value("LambNoBias", AdamMode::LambNoBias);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
