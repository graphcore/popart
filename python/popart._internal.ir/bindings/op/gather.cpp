// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/gather.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep

#include "popart/op.hpp"
#include "popart/op/gather.hpp"

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindGather(py::module &m) {
  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<GatherOp, popart::Op, std::shared_ptr<GatherOp>>(sm, "GatherOp")
      .def(py::init<const OperatorIdentifier &,
                    int64_t,
                    const Op::Settings &,
                    const nonstd::optional<float> &,
                    bool>(),
           py::arg("opid"),
           py::arg("axis"),
           py::arg("settings"),
           py::arg("available_memory_proportion_"),
           py::arg("zeroOutOfRangeIndices_"))
      .def("getAvailableMemoryProportion",
           &GatherOp::getAvailableMemoryProportion)
      .def("setAvailableMemoryProportion",
           &GatherOp::setAvailableMemoryProportion);

  py::class_<GatherGradOp, popart::Op, std::shared_ptr<GatherGradOp>>(
      sm, "GatherGradOp")
      .def("getAvailableMemoryProportion",
           &GatherGradOp::getAvailableMemoryProportion)
      .def("setAvailableMemoryProportion",
           &GatherGradOp::setAvailableMemoryProportion);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
