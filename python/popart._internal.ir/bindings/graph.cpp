// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/graph.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../popart/shared_cpp/np_utils.hpp"
#include <popart/graph.hpp>
#include <popart/graphid.hpp>
#include <popart/ir.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindGraph(py::module &m) {
  py::class_<Graph>(m, "Graph")
      .def(py::init<Ir &, const GraphId &>())
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
      .def("getTensor", &Graph::getTensor, py::return_value_policy::reference)
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
      .def("getGraphString", &Graph::getGraphString);
}

} // namespace ir
} // namespace _internal
} // namespace popart
