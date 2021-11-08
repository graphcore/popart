// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/ir.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindIr(py::module &m) {

  py::class_<Ir, std::shared_ptr<Ir>>(m, "Ir")
      .def(py::init<>())
      .def("getId", &Ir::getId)
      .def("getMainGraph",
           py::overload_cast<>(&Ir::getMainGraph),
           py::return_value_policy::reference)
      .def("getAllGraphs",
           &popart::Ir::getAllGraphs,
           py::return_value_policy::reference)
      .def(
          "getGraph", &popart::Ir::getGraph, py::return_value_policy::reference)
      .def("getSessionOptions",
           py::overload_cast<>(&popart::Ir::getSessionOptions),
           py::return_value_policy::reference)
      .def("hasGraph", &popart::Ir::hasGraph)
      .def("createGraph",
           &popart::Ir::createGraph,
           py::return_value_policy::reference)
      .def("createUniqueSubgraphId",
           [](Ir &self, const std::string &id) -> std::string {
             return self.createUniqueSubgraphId(GraphId{id}).str();
           })
      .def("removeGraph", &popart::Ir::removeGraph)
      .def("setIsPrepared", &Ir::setIsPrepared)
      .def("isPrepared", &Ir::isPrepared)
      .def("setDataFlow", &Ir::setDataFlow)
      .def("getDataFlow", &Ir::getDataFlow)
      .def("applyTransform", &popart::Ir::applyTransform)
      .def("setDeviceInfo", &Ir::setDeviceInfo)
      .def("logIr", &Ir::logIr)
      .def("createIntermediateTensorId", &Ir::createIntermediateTensorId)
      .def("serializeToJSON",
           [](Ir &self) {
             std::stringstream ss;
             self.serialise(Ir::SerialiseFormat::JSON, ss);
             return ss.str();
           })
      .def("updateVertices", &Ir::updateVertices)
      .def("setPatterns", &Ir::setPatterns)
      .def("applyPreAliasPatterns", &Ir::applyPreAliasPatterns)
      .def("applyInplacePattern", &Ir::applyInplacePattern)
      .def("setRemoteBufferInfo", &Ir::setRemoteBufferInfo)
      .def("dataStreamTensors",
           &Ir::dataStreamTensors,
           py::return_value_policy::reference);
}

} // namespace ir
} // namespace _internal
} // namespace popart
