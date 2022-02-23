// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "bindings/tensor.hpp"
#include "../../popart/shared_cpp/np_utils.hpp"

#include "popart/debugcontext.hpp"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <popart/aliasesmap.hpp>
#include <popart/graph.hpp>
#include <popart/half.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/tensor.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

namespace {
/**
 * Get the TensorData from a tensor. See Devicex::remoteBufferWeightsToHost.
 *
 * @tparam RESULT_TYPE The data type of the tensor.
 * \param t The tensor.
 * \returns py::array A Python array of data.
 */
template <typename RESULT_TYPE> py::array getTensorData(Tensor &t) {
  auto data = reinterpret_cast<RESULT_TYPE *>(t.tensorData()->data());
  std::vector<int64_t> shape = t.info.shape();
  auto strides               = t.info.strides();
  // Note: py::memoryview::from_buffer doesn't malloc, it just provides a view
  // into some existing memory
  // See:
  // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#memory-view
  return py::memoryview::from_buffer(data, // buffer pointer
                                     shape,
                                     strides);
}
} // namespace

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

  py::class_<Consumers>(m, "Consumers")
      .def("getOps", &Consumers::getOps, py::return_value_policy::reference);

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
      .def("hasProducer", &Tensor::hasProducer)
      .def("getProducer",
           &Tensor::getProducer,
           py::return_value_policy::reference)
      // .def("setProducer", &Tensor::setProducer, py::arg("op"))
      // .def("resetProducer", &Tensor::resetProducer, py::arg("op"))
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
      // tensorData defined for different data types below
      .def("anyAlias", &Tensor::anyAlias)
      .def("setTensorData",
           [](Tensor &self, const TensorInfo &info, py::array data) {
             data      = makeContiguous(data);
             self.info = info;
             self.setTensorData(info, data.request().ptr);
           })
      .def("associatedOps", &Tensor::associatedOps)
      .def("getGraph",
           py::overload_cast<>(&Tensor::getGraph),
           py::return_value_policy::reference)
      .def("getGraph_const",
           py::overload_cast<>(&Tensor::getGraph, py::const_),
           py::return_value_policy::reference)
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
           static_cast<VGraphIdAndTileSet (Tensor::*)() const>(
               &Tensor::getVirtualGraphIdAndTileSetUnsafe))
      .def("getVirtualGraphIdAndTileSetUnsafe",
           static_cast<VGraphIdAndTileSet (Tensor::*)(std::set<OpId> &) const>(
               &Tensor::getVirtualGraphIdAndTileSetUnsafe))
      .def("getBatchAxis", &Tensor::getBatchAxis)
      .def("consumersAllPreLoss", &Tensor::consumersAllPreLoss)
      .def("isModified", &Tensor::isModified)
      .def("isAliased", &Tensor::isAliased)
      .def("modifiedRegionsByOps",
           [](Tensor &self, std::vector<Op *> ops) {
             auto &graph = self.getGraph();
             // Not binding Aliases/AliasesMap for now.
             AliasesMap aliasMap(graph);
             return self.modifiedRegionsByOps(ops, aliasMap.getAliases(graph));
           })
      .def("getDataViaGraphTraversal", &Tensor::getDataViaGraphTraversal)
      .def("getDebugInfo",
           &Tensor::getDebugInfo,
           py::return_value_policy::reference)
      .def("getCopyFromTensor", &Tensor::getCopyFromTensor)
      .def("getVariableUpdateType", &Tensor::getVariableUpdateType)
      .def("setCopyFromTensor", &Tensor::setCopyFromTensor, py::arg("value"))
      .def("setVariableUpdateType",
           &Tensor::setVariableUpdateType,
           py::arg("type"))
      .def_readonly("consumers", &Tensor::consumers)
      .def_readwrite("info", &Tensor::info)
      .def_readonly("tensorLocationInfo", &Tensor::tensorLocationInfo)
      .def_readonly("inputSettings", &Tensor::inputSettings)
      .def_readonly("id", &Tensor::id)
      .def("isInSyncWithIPU",
           [](Tensor &self) { return self.tensorData()->getIsSyncedWithIPU(); })
      .def("dataAsFloat64",
           [](Tensor &self) { return getTensorData<double>(self); })
      .def("dataAsFloat32",
           [](Tensor &self) { return getTensorData<float>(self); })
      .def("dataAsFloat16",
           // TODO T50782: Handle fp16 conversion in cpp and avoid the .view()
           // call in python. We use uint16_t as it's how float16_t is stored in
           // popart.
           [](Tensor &self) { return getTensorData<uint16_t>(self); })
      .def("dataAsInt64",
           [](Tensor &self) { return getTensorData<int64_t>(self); })
      .def("dataAsInt32",
           [](Tensor &self) { return getTensorData<int32_t>(self); })
      .def("dataAsInt16",
           [](Tensor &self) { return getTensorData<int16_t>(self); })
      .def("dataAsInt8",
           [](Tensor &self) { return getTensorData<int8_t>(self); })
      .def("dataAsUInt64",
           [](Tensor &self) { return getTensorData<uint64_t>(self); })
      .def("dataAsUInt32",
           [](Tensor &self) { return getTensorData<uint32_t>(self); })
      .def("dataAsUInt16",
           [](Tensor &self) { return getTensorData<uint16_t>(self); })
      .def("dataAsBool", [](Tensor &self) { return getTensorData<bool>(self); })
      .def(
          "writeTensorData",
          [](Tensor &self, py::array &npArray) {
            if (!isContiguous(npArray)) {
              throw error(
                  "writeToMemory is unable to use the numpy output array for "
                  "tensor "
                  "'{}' as it is not c-contiguous (a data conversion here "
                  "could have a "
                  "significant impact on performance and hence is not allowed)",
                  self.id);
            }
            self.tensorData()->resetData(
                self.info, static_cast<void *>(npArray.request().ptr));
          })
      .def("setTensorLocationInfo",
           [](Tensor &self, TensorLocation &tLocation, int a, int b) {
             auto pair = std::pair<RemoteBufferId, RemoteBufferIndex>(a, b);
             self.setTensorLocationInfo(tLocation, pair);
           });

  py::class_<TensorLocationInfo>(m, "TensorLocationInfo")
      .def("setRemoteBufferInfo", &TensorLocationInfo::setRemoteBufferInfo)
      .def("getRemoteBufferInfo", &TensorLocationInfo::getRemoteBufferInfo)
      .def("isRemote", &TensorLocationInfo::isRemote)
      .def("isSharded", &TensorLocationInfo::isSharded);
}

} // namespace ir
} // namespace _internal
} // namespace popart
