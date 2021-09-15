// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/graph.hpp"
#include "bindings/op.hpp"

#include <algorithm>
#include <iterator>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../popart/shared_cpp/np_utils.hpp"
#include <popart/graph.hpp>
#include <popart/graphid.hpp>
#include <popart/ir.hpp>
#include <popart/op/call.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindGraph(py::module &m) {
  py::class_<Graph> g(m, "Graph");

  g.def(py::init<Ir &, const GraphId &>())
      .def("addActGrad", &Graph::addActGrad)
      .def(
          "addVarInit",
          [](Graph &self,
             const TensorId &tid,
             const TensorInfo &tinfo,
             py::array data,
             const DebugContext &dc) {
            data = makeContiguous(data);
            self.addVarInit(tid, tinfo, data.request().ptr, dc);
          },
          py::arg("tensorId"),
          py::arg("tensorInfo"),
          py::arg("data"),
          py::arg("debugContext") = std::string())
      .def(
          "addConstInit",
          [](Graph &self,
             const TensorId &tid,
             const TensorInfo &tinfo,
             py::array data,
             const DebugContext &dc) {
            data = makeContiguous(data);
            self.addConstInit(tid, tinfo, data.request().ptr, dc);
          },
          py::arg("tensorId"),
          py::arg("tensorInfo"),
          py::arg("data"),
          py::arg("debugContext") = std::string())
      .def("addStream", &Graph::addStream)
      .def("getTensor", &Graph::getTensor, py::return_value_policy::reference)
      .def(
          "getTensors",
          [](Graph &self) { return self.getTensors().getAll(); },
          py::return_value_policy::reference)
      .def(
          "getTensorsOfType",
          [](Graph &self, TensorType tensor_type) {
            return self.getTensors().getOfType(tensor_type);
          },
          py::return_value_policy::reference)
      .def("getInputIds",
           &Graph::getInputIds,
           py::return_value_policy::reference)
      .def("getInputIndex", &Graph::getInputIndex)
      .def("addInput",
           py::overload_cast<const InIndex &,
                             const TensorId &,
                             const TensorInfo &,
                             bool>(&Graph::addInput))
      .def("addInput",
           py::overload_cast<const TensorId &, const TensorInfo &>(
               &Graph::addInput))
      .def("getInputId", &Graph::getInputId)
      .def("hasInputId", &Graph::hasInputId)
      .def("removeInput",
           py::overload_cast<const TensorId &>(&Graph::removeInput))
      .def("removeInput",
           py::overload_cast<const InIndex &>(&Graph::removeInput))
      .def("getOutputIds",
           &Graph::getOutputIds,
           py::return_value_policy::reference)
      .def("getOutputIndex", &Graph::getOutputIndex)
      .def("markAsOutput",
           py::overload_cast<const OutIndex &, const TensorId &, bool>(
               &Graph::markAsOutput))
      .def("markAsOutput",
           py::overload_cast<const TensorId &>(&Graph::markAsOutput))
      .def("removeOutput",
           py::overload_cast<const TensorId &>(&Graph::removeOutput))
      .def("removeOutput",
           py::overload_cast<const OutIndex &>(&Graph::removeOutput))
      .def("getOutputId", &Graph::getOutputId)
      .def("hasOutputId", &Graph::hasOutputId)
      .def("addScope", &Graph::addScope)
      .def("removeScope", &Graph::removeScope)
      .def("getScope", &Graph::getScope)
      .def_readonly("id", &Graph::id)
      .def("getGraphString", &Graph::getGraphString)
      .def("getOpIds", &Graph::getOpIds)
      .def("__contains__",
           [](Graph &self, const TensorId &name) {
             return self.getTensors().contains(name);
           })
      .def("getIr", py::overload_cast<>(&Graph::getIr, py::const_))
      .def(
          "getOps",
          [](const Graph &self) -> std::vector<Op *> {
            const auto nOps = self.getOps().size();
            std::vector<Op *> ops;
            ops.reserve(nOps);

            std::transform(self.getOps().cbegin(),
                           self.getOps().cend(),
                           std::back_inserter(ops),
                           [](auto &id_op) { return id_op.second.get(); });
            return ops;
          },
          py::return_value_policy::reference);

  bindCreateOpFunctionToGraphClass(g);

  bindCreateConnectedOpFunctionToGraphClass(g);
}

} // namespace ir
} // namespace _internal
} // namespace popart
