// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/call.hpp"
#include "bindings/op.hpp"
#include "bindings/op/optional.hpp"

#include "bindings/basicoptionals.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/graph.hpp>
#include <popart/op/call.hpp>
#include <popart/region.hpp>
#include <popart/vendored/optional.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindCall(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<CallOp, popart::Op, std::shared_ptr<CallOp>>(sm, "CallOp")
      .def(py::init<const popart::OperatorIdentifier &,
                    popart::Graph &,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("callee"),
           py::arg("settings"))
      .def(py::init<const popart::OperatorIdentifier &,
                    popart::Graph &,
                    const std::vector<int> &,
                    const Op::Settings &>(),
           py::arg("opid"),
           py::arg("callee"),
           py::arg("modifiedInputsViaAttrs"),
           py::arg("settings"))
      .def("subgraphInToOpInIndex", &CallOp::subgraphInToOpInIndex)
      .def("opInToSubgraphInIndex", &CallOp::opInToSubgraphInIndex)
      .def("subgraphOutToOpOutIndex", &CallOp::subgraphOutToOpOutIndex)
      .def("opOutToSubgraphOutIndex", &CallOp::opOutToSubgraphOutIndex)
      .def("addModified", &CallOp::addModified)
      .def("getCalledGraph",
           &CallOp::getCalledGraph,
           py::return_value_policy::reference);

  py::class_<CallGradOp, popart::Op, std::shared_ptr<CallGradOp>>(sm,
                                                                  "CallGradOp")
      .def(py::init<popart::CallOp &,
                    popart::Graph &,
                    const std::vector<GradInOutMapper> &,
                    const std::map<int, int> &>(),
           py::arg("fwdOp"),
           py::arg("bwdGraph"),
           py::arg("gradInInfo_"),
           py::arg("gradOutInfo_"));
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
