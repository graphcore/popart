// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/accumulatorzero.hpp"
#include "bindings/op.hpp"
#include "bindings/op/optional.hpp"
#include "bindings/op/varupdate.hpp"

#include "bindings/basicoptionals.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/op/accumulatorzero.hpp>
#include <popart/vendored/optional.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

void bindAccumulatorZero(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<AccumulatorZeroOp,
             AccumulatorScaleOp,
             PyVarUpdateOp<AccumulatorZeroOp>,
             std::shared_ptr<AccumulatorZeroOp>>(sm, "AccumulatorZeroOp")
      .def(py::init<const Op::Settings &>(), py::arg("settings"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart