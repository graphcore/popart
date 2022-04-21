// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "bindings/op/roialign.hpp"
#include "bindings/op.hpp"
#include "bindings/op/optional.hpp"

#include "bindings/basicoptionals.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace popart {
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
