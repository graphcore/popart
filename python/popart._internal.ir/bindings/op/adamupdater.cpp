// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/adamupdater.hpp"
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

void bindAdamUpdater(py::module &m) {
  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<AdamUpdaterOp, popart::Op, std::shared_ptr<AdamUpdaterOp>>(
      sm, "AdamUpdaterOp")
      .def(py::init<AdamMode,
                    OptimizerValue,
                    OptimizerValue,
                    OptimizerValue,
                    OptimizerValue,
                    const Op::Settings &>(),
           py::arg("mode_"),
           py::arg("wd"),
           py::arg("b1"),
           py::arg("b2"),
           py::arg("eps"),
           py::arg("settings"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart