// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "bindings/op/if.hpp"

#include <initializer_list>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>     // IWYU pragma: keep
#include <pybind11/operators.h> // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <popart/op/if.hpp>

#include "popart/graph.hpp" // IWYU pragma: keep
#include "popart/names.hpp"
#include "popart/op.hpp"

namespace py = pybind11;

namespace popart {
struct OperatorIdentifier;

namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindIf(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<BranchInfo>(sm, "BranchInfo")
      .def(py::init<const GraphId &,
                    const std::map<int, int>,
                    const std::map<int, int>>(),
           py::arg("graphId"),
           py::arg("inputIndicesMap"),
           py::arg("outputIndicesMap"));

  py::class_<IfOp, popart::Op, std::shared_ptr<IfOp>>(sm, "IfOp")
      .def(py::init<const popart::OperatorIdentifier &,
                    const BranchInfo &,
                    const BranchInfo &,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("thenBranchInfo"),
           py::arg("elseBranchInfo"),
           py::arg("settings"))
      .def_static("getConditionInIndex",
                  static_cast<InIndex (*)()>(&IfOp::getConditionInIndex));

  py::class_<IfGradOp, IfOp, std::shared_ptr<IfGradOp>>(sm, "IfGradOp")
      .def(py::init<const IfOp &,
                    const std::vector<GradInOutMapper> &,
                    const BranchInfo &,
                    const BranchInfo &>(),
           py::arg("fwdOp"),
           py::arg("gradInInfo_"),
           py::arg("thenBranchInfo"),
           py::arg("elseBranchInfo"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
