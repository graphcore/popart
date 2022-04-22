// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/accumulatorscale.hpp"

#include <array>
#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/op/accumulatorscale.hpp>

#include "bindings/op/varupdate.hpp"
#include "popart/alias/aliasmodel.hpp" // IWYU pragma: keep
#include "popart/op.hpp"

namespace py = pybind11;

namespace popart {
class OptimizerValue;
class VarUpdateOp;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindAccumulatorScale(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<AccumulatorScaleOp,
             VarUpdateOp,
             PyVarUpdateOp<AccumulatorScaleOp>,
             std::shared_ptr<AccumulatorScaleOp>>(sm, "AccumulatorScaleOp")
      .def(py::init<const OptimizerValue, const Op::Settings &>(),
           py::arg("factor"),
           py::arg("settings"))
      .def("getFactor", &AccumulatorScaleOp::getFactor);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
