// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/op/loop.hpp"
#include "bindings/op.hpp"
#include "bindings/op/optional.hpp"

#include "bindings/basicoptionals.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/graph.hpp>
#include <popart/op/loop.hpp>
#include <popart/region.hpp>
#include <popart/vendored/optional.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {
namespace op {

// cppcheck-suppress constParameter // False positive for &m
void bindRepeat(py::module &m) {

  auto sm = m;

  sm = sm.def_submodule("op", "Python bindings for PopART ops.");

  py::class_<LoopOp, popart::Op, std::shared_ptr<LoopOp>>(sm, "LoopOp")
      .def(py::init<const popart::OperatorIdentifier &,
                    const Op::Settings &,
                    popart::Graph &>(),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("callee"))
      .def(py::init<const popart::OperatorIdentifier &,
                    const Op::Settings &,
                    popart::Graph &,
                    int &>(),
           py::arg("opid"),
           py::arg("settings"),
           py::arg("callee"),
           py::arg("numImplicitScanOutputs"))
      .def("addLoopInput", &LoopOp::addLoopInput)
      .def("addLoopOutput", &LoopOp::addLoopOutput)
      .def("removeLoopInput", &LoopOp::removeLoopInput)
      .def("removeLoopOutput", &LoopOp::removeLoopOutput)
      .def("getTripCountValue", &LoopOp::getTripCountValue)
      .def("setTripCountValue", &LoopOp::setTripCountValue)
      .def("getCalledGraph", &LoopOp::getCalledGraph)
      .def("opInToSubgraphInIndex",
           py::overload_cast<OutIndex>(&LoopOp::opInToSubgraphInIndex,
                                       py::const_))
      .def("opOutToSubgraphOutIndex",
           py::overload_cast<OutIndex>(&LoopOp::opOutToSubgraphOutIndex,
                                       py::const_))
      .def("subgraphInToOpInIndex",
           py::overload_cast<InIndex>(&LoopOp::subgraphInToOpInIndex,
                                      py::const_))
      .def("subgraphOutToOpOutIndex",
           py::overload_cast<InIndex>(&LoopOp::subgraphOutToOpOutIndex,
                                      py::const_))
      .def("addModified", &LoopOp::addModified)
      .def("getNumExplicitInputs", &LoopOp::getNumExplicitInputs);
}
} // namespace op
} // namespace ir
} // namespace _internal
} // namespace popart
