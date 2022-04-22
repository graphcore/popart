// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/adamupdater.hpp"

#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include "popart/adam.hpp"
#include "popart/op.hpp"
#include "popart/op/adamupdater.hpp"

namespace py = pybind11;

namespace popart {
class OptimizerValue;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
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
