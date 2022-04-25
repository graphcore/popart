// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/graph.hpp"

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // ITYU pragma: keep
#include <set>
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/graphid.hpp>

#include "../../popart/shared_cpp/np_utils.hpp"
#include "bindings/op/manualbindops.hpp"
#include "popart/debugcontext.hpp"
#include "popart/ir.hpp" // IWYU pragma: keep
#include "popart/names.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensors.hpp"
#include "popart/topocons.hpp" // IWYU pragma: keep
#include "popart/variablesettings.hpp"

namespace py = pybind11;

namespace popart {
class Op;
class TensorInfo;

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
          "addVarInit",
          [](Graph &self,
             const TensorId &tid,
             const TensorInfo &tinfo,
             py::array data,
             const VariableSettings &vs,
             const DebugContext &dc) {
            data = makeContiguous(data);
            self.addVarInit(tid, tinfo, data.request().ptr, vs, dc);
          },
          py::arg("tensorId"),
          py::arg("tensorInfo"),
          py::arg("data"),
          py::arg("vs"),
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
      .def("getInputTensor",
           &Graph::getInputTensor,
           py::return_value_policy::reference)
      .def("getOutputTensor",
           &Graph::getOutputTensor,
           py::return_value_policy::reference)
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
      .def("getScope", &Graph::getScope)
      .def_readonly("id", &Graph::id)
      .def("getGraphString", &Graph::getGraphString)
      .def("getOpIds", &Graph::getOpIds)
      .def("__contains__",
           [](Graph &self, const TensorId &name) {
             return self.getTensors().contains(name);
           })
      .def("getIr",
           py::overload_cast<>(&Graph::getIr, py::const_),
           py::return_value_policy::reference)
      .def(
          "topoCons",
          [](Graph &self) { return self.topoCons.get(); },
          py::return_value_policy::reference)
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
          py::return_value_policy::reference)
      .def(
          "getOpSchedule",
          [](const Graph &self,
             bool requireOptimalSchedule) -> std::vector<Op *> {
            return self.getOpSchedule({},
                                      requireOptimalSchedule
                                          ? RequireOptimalSchedule::Yes
                                          : RequireOptimalSchedule::No);
          },
          py::arg("requireOptimalSchedule") = true,
          py::return_value_policy::reference)
      .def("getOp", &Graph::getOp, py::return_value_policy::reference)
      .def("eraseOp", [](Graph &self, OpId opid) { self.eraseOp(opid); })
      .def("getCalledGraphs",
           &Graph::getCalledGraphs,
           py::return_value_policy::reference)
      .def("getAllVirtualGraphIds", &Graph::getAllVirtualGraphIds)
      .def("removeIsolatedTensors",
           &Graph::removeIsolatedTensors,
           py::arg("retainUsedIOTensors") = false,
           py::arg("retainAllIOTensors")  = false,
           py::arg("retainVarTensors")    = false,
           py::arg("retainConstTensors")  = false);

  bindCreateOpFunctionToGraphClass(g);

  bindCreateConnectedOpFunctionToGraphClass(g);

  bindManualCreateOpFunctionToGraphClass(g);

  bindManualCreateConnectedOpFunctionToGraphClass(g);
}

} // namespace ir
} // namespace _internal
} // namespace popart
