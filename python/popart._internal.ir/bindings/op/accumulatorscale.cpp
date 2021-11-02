// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/accumulatorscale.hpp"
#include "bindings/op.hpp"
#include "bindings/op/optional.hpp"
#include "bindings/op/varupdate.hpp"

#include "bindings/basicoptionals.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/op/accumulatorscale.hpp>
#include <popart/vendored/optional.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

void bindAccumulatorScale(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<AccumulatorScaleOp,
             VarUpdateOp,
             PyVarUpdateOp<AccumulatorScaleOp>,
             std::shared_ptr<AccumulatorScaleOp>>(sm, "AccumulatorScaleOp")
      .def(py::init<const OptimizerValue, const Op::Settings &>(),
           py::arg("factor"),
           py::arg("settings"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart