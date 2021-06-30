// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/tensor.hpp"
#include "../../popart/shared_cpp/np_utils.hpp"

#include "popart/debugcontext.hpp"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/tensor.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

void bindTensor(py::module &m) {
  py::enum_<TensorType>(m, "TensorType")
      .value("ActGrad", TensorType::ActGrad)
      .value("Const", TensorType::Const)
      .value("Stream", TensorType::Stream)
      .value("Unknown", TensorType::Unknown)
      .value("Variable", TensorType::Variable)
      .value("N", TensorType::N);

  py::enum_<VariableUpdateType>(m, "VariableUpdateType")
      .value("None_", VariableUpdateType::None)
      .value("Gradient", VariableUpdateType::Gradient)
      .value("Copy", VariableUpdateType::Copy);

  py::class_<TensorTypeInfo>(m, "TensorTypeInfo")
      .def(py::init<TensorType, std::string>(), py::arg("t_"), py::arg("ts_"))
      .def("type", &TensorTypeInfo::type)
      .def("type_s",
           &TensorTypeInfo::type_s,
           py::return_value_policy::reference);

  py::class_<Tensor>(m, "Tensor")
      .def(py::init<TensorId, TensorType, Graph &>(),
           py::arg("tensorId"),
           py::arg("tensorType"),
           py::arg("graph"))
      .def(py::init<TensorId, TensorType, Graph &, const DebugContext &>(),
           py::arg("tensorId"),
           py::arg("tensorType"),
           py::arg("graph"),
           py::arg("debugContext"))
      .def("str", &Tensor::str)
      .def("clone", &Tensor::clone, py::arg("graph"))
      .def("tensorType", &Tensor::tensorType)
      .def("tensor_type", &Tensor::tensor_type)
      .def("getTensorTypeInfo",
           &Tensor::getTensorTypeInfo,
           py::return_value_policy::reference)
      .def("setTensorType", &Tensor::setTensorType, py::arg("tensorType"))
      .def("getReplicatedStreamMode", &Tensor::getReplicatedStreamMode)
      .def("setReplicatedStreamMode",
           &Tensor::setReplicatedStreamMode,
           py::arg("mode"))
      // TODO(T42230): Bind and test getPipelineStages.
      // .def("getPipelineStages", &Tensor::getPipelineStages)
      // TODO(T42234): Bind and test getProducerUnsafe, getProducer, setProducer
      // resetProducer, hasProducer.
      // .def("getProducerUnsafe", &Tensor::getProducerUnsafe)
      // .def("getProducer", &Tensor::getProducer)
      // .def("setProducer", &Tensor::setProducer, py::arg("op"))
      // .def("resetProducer", &Tensor::resetProducer, py::arg("op"))
      // .def("hasProducer", &Tensor::hasProducer)
      // TODO(T42233): Bind and test isGraphInput, getGraphInputIndex,
      // isGraphOutput, getGraphOutputIndex.
      // .def("isGraphInput", &Tensor::isGraphInput)
      // .def("getGraphInputIndex", &Tensor::getGraphInputIndex)
      // .def("isGraphOutput", &Tensor::isGraphOutput)
      // .def("getGraphOutputIndex", &Tensor::getGraphOutputIndex)
      // TODO(T42234): Bind and test isLoopInput, isImplicitLoopInput,
      // isExplicitLoopInput, isLoopTripCounter.
      // .def("isLoopInput", &Tensor::isLoopInput)
      // .def("isImplicitLoopInput", &Tensor::isImplicitLoopInput)
      // .def("isExplicitLoopInput", &Tensor::isExplicitLoopInput)
      // .def("isLoopTripCounter", &Tensor::isLoopTripCounter)
      .def("isUnmodifiable", &Tensor::isUnmodifiable)
      .def("isCheckpointTensor", &Tensor::isCheckpointTensor)
      .def("isImplicitRecomputeTensor", &Tensor::isImplicitRecomputeTensor)
      .def("isRestoreInplaceTensor", &Tensor::isRestoreInplaceTensor)
      .def("idIncludesPrefix", &Tensor::idIncludesPrefix)
      .def("isOptimizerTensor", &Tensor::isOptimizerTensor)
      .def("isRemoteArgTensor", &Tensor::isRemoteArgTensor)
      .def("isRandomSeedTensor", &Tensor::isRandomSeedTensor)
      .def("isOptimizerStateTensor", &Tensor::isOptimizerStateTensor)
      .def("isAccumulatorTensor", &Tensor::isAccumulatorTensor)
      .def("isHostLoadTensor", &Tensor::isHostLoadTensor)
      .def("isWeightTensor", &Tensor::isWeightTensor)
      .def("isAnchored", &Tensor::isAnchored)
      .def("isRootAnchor", &Tensor::isRootAnchor)
      .def("hasTensorData", &Tensor::hasTensorData)
      .def("tensorData",
           py::overload_cast<>(&Tensor::tensorData),
           py::return_value_policy::reference)
      .def("tensorData_const",
           py::overload_cast<>(&Tensor::tensorData, py::const_),
           py::return_value_policy::reference)
      .def("anyAlias", &Tensor::anyAlias)
      .def("setTensorData",
           [](Tensor &self, const TensorInfo &info, py::array data) {
             data = makeContiguous(data);
             self.setTensorData(info, data.request().ptr);
           })
      .def("associatedOps", &Tensor::associatedOps)
      .def("getGraph", py::overload_cast<>(&Tensor::getGraph))
      .def("getGraph_const", py::overload_cast<>(&Tensor::getGraph, py::const_))
      .def("getIr",
           py::overload_cast<>(&Tensor::getIr),
           py::return_value_policy::reference)
      .def("getIr_const",
           py::overload_cast<>(&Tensor::getIr, py::const_),
           py::return_value_policy::reference)
      .def("hasVirtualGraphId", &Tensor::hasVirtualGraphId)
      .def("getVirtualGraphId", &Tensor::getVirtualGraphId)
      .def("getVirtualGraphIdUnsafe", &Tensor::getVirtualGraphIdUnsafe)
      .def("getVirtualGraphIdAndTileSet", &Tensor::getVirtualGraphIdAndTileSet)
      .def("getVirtualGraphIdAndTileSetUnsafe",
           &Tensor::getVirtualGraphIdAndTileSetUnsafe)
      .def("getBatchAxis", &Tensor::getBatchAxis)
      .def("consumersAllPreLoss", &Tensor::consumersAllPreLoss)
      .def("isModified", &Tensor::isModified)
      .def("isAliased", &Tensor::isAliased)
      // TODO(T42224): Enable and test the following. Needs binding of Aliases.
      // .def("modifiedRegionsByOps",
      //      py::overload_cast<std::vector<Op *>, Aliases &>(
      //          &Tensor::modifiedRegionsByOps),
      //      py::arg("ops"),
      //      py::arg("aliases"))
      // .def("modifiedRegionsByOps",
      //      py::overload_cast<std::vector<OpId>, Aliases &>(
      //          &Tensor::modifiedRegionsByOps),
      //      py::arg("opIds"),
      //      py::arg("aliases"))
      .def("getDataViaGraphTraversal", &Tensor::getDataViaGraphTraversal)
      .def("getDebugInfo",
           &Tensor::getDebugInfo,
           py::return_value_policy::reference)
      .def_readonly("consumers", &Tensor::consumers)
      .def_readonly("info", &Tensor::info)
      .def_readonly("tensorLocationInfo", &Tensor::tensorLocationInfo)
      .def_readonly("inputSettings", &Tensor::inputSettings)
      .def_readonly("id", &Tensor::id);

  py::enum_<Tensor::ReplicatedStreamMode>(m.attr("Tensor"),
                                          "ReplicatedStreamMode")
      .value("Replicate", Tensor::ReplicatedStreamMode::Replicate)
      .value("Broadcast", Tensor::ReplicatedStreamMode::Broadcast);
}

} // namespace ir
} // namespace _internal
} // namespace popart
