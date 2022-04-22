// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "bindings/op/roialign.hpp"

#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <stdint.h>

#include "popart/op.hpp"
#include "popart/op/roialign.hpp"

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindRoiAlign(py::module &m) {
  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<RoiAlignOp, Op, std::shared_ptr<RoiAlignOp>>(sm, "RoiAlignOp")
      .def(py::init<const OperatorIdentifier &,
                    const Op::Settings &,
                    const float,
                    const uint64_t,
                    const uint64_t,
                    const uint64_t>(),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("spatialScale"),
           py::arg("samplingRatio"),
           py::arg("alignedHeight"),
           py::arg("alignedWidth"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
