// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/accumulatorzero.hpp"

#include <array>
#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/op/accumulatorzero.hpp>

#include "bindings/op/varupdate.hpp"
#include "popart/alias/aliasmodel.hpp" // IWYU pragma: keep
#include "popart/op.hpp"

namespace py = pybind11;

namespace popart {
class AccumulatorScaleOp;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
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
