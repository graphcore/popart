// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "executablexserializer.hpp"

#include <algorithm>
#include <capnp/blob.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <cstdint>
#include <cstdlib>
#include <gcl/CollectiveBalancedReorder.hpp>
#include <kj/common.h>
#include <kj/std/iostream.h>
#include <map>
#include <memory>
#include <onnxutil.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <capnp/blob.h>
#include <capnp/serialize.h>

#include <kj/common.h>
#include <kj/std/iostream.h>

#include <gcl/CollectiveBalancedReorder.hpp>

#include <popef/Reader.hpp>
#include <popef/Types.hpp>
#include <popef/Writer.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Executable.hpp>
#include <poplar/Interval.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/StringRef.hpp>
#include <poplar/Target.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/variablesettings.hpp>
#include <popart/voiddata.hpp>

#include <onnxutil.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/capnp/Executablex.capnp.h"
#include "popart/capnp/Ir.capnp.h"
#include "popart/capnp/IrLowering.capnp.h"
#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/replicatedtensorshardingbundle.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/util.hpp"
#include "popart/voiddata.hpp"
#include <popart/vendored/any.hpp> // IWYU pragma: keep

namespace popart {
namespace popx {
namespace serialization {

popart::cap::TensorType toCapnpTensorType(popart::TensorType type) {
  switch (type) {
  case popart::TensorType::ActGrad:
    return popart::cap::TensorType::ACT_GRAD;
  case popart::TensorType::Const:
    return popart::cap::TensorType::CONSTANT;
  case popart::TensorType::Stream:
    return popart::cap::TensorType::STREAM;
  case popart::TensorType::Unknown:
    return popart::cap::TensorType::UNKNOWN;
  case popart::TensorType::Variable:
    return popart::cap::TensorType::VARIABLE;
  case popart::TensorType::N:
    return popart::cap::TensorType::N;
  }

  std::stringstream errorStream;
  errorStream << "Invalid TensorType " << type;
  throw error(errorStream.str());
}

popart::TensorType toPopartTensorType(popart::cap::TensorType type) {
  switch (type) {
  case popart::cap::TensorType::ACT_GRAD:
    return popart::TensorType::ActGrad;
  case popart::cap::TensorType::CONSTANT:
    return popart::TensorType::Const;
  case popart::cap::TensorType::STREAM:
    return popart::TensorType::Stream;
  case popart::cap::TensorType::UNKNOWN:
    return popart::TensorType::Unknown;
  case popart::cap::TensorType::VARIABLE:
    return popart::TensorType::Variable;
  case popart::cap::TensorType::N:
    return popart::TensorType::N;
  }

  std::stringstream errorStream;
  errorStream << "Invalid TensorType " << type;
  throw error(errorStream.str());
}

popart::cap::DataType toCapnpDataType(popart::DataType type) {
  switch (type) {
  case popart::DataType::UINT8:
    return popart::cap::DataType::UINT8;
  case popart::DataType::INT8:
    return popart::cap::DataType::INT8;
  case popart::DataType::UINT16:
    return popart::cap::DataType::UINT16;
  case popart::DataType::INT16:
    return popart::cap::DataType::INT16;
  case popart::DataType::INT32:
    return popart::cap::DataType::INT32;
  case popart::DataType::INT64:
    return popart::cap::DataType::INT64;
  case popart::DataType::UINT32:
    return popart::cap::DataType::UINT32;
  case popart::DataType::UINT64:
    return popart::cap::DataType::UINT64;
  case popart::DataType::BOOL:
    return popart::cap::DataType::BOOL;
  case popart::DataType::FLOAT:
    return popart::cap::DataType::FLOAT;
  case popart::DataType::FLOAT16:
    return popart::cap::DataType::FLOAT16;
  case popart::DataType::BFLOAT16:
    return popart::cap::DataType::BFLOAT16;
  case popart::DataType::DOUBLE:
    return popart::cap::DataType::DOUBLE;
  case popart::DataType::COMPLEX64:
    return popart::cap::DataType::COMPLEX64;
  case popart::DataType::COMPLEX128:
    return popart::cap::DataType::COMPLEX128;
  case popart::DataType::STRING:
    return popart::cap::DataType::STRING;
  case popart::DataType::UNDEFINED:
    return popart::cap::DataType::UNDEFINED;
  }

  std::stringstream errorStream;
  errorStream << "Invalid DataType " << type;
  throw error(errorStream.str());
}

popart::DataType toPopartDataType(popart::cap::DataType type) {
  switch (type) {
  case popart::cap::DataType::UINT8:
    return popart::DataType::UINT8;
  case popart::cap::DataType::INT8:
    return popart::DataType::INT8;
  case popart::cap::DataType::UINT16:
    return popart::DataType::UINT16;
  case popart::cap::DataType::INT16:
    return popart::DataType::INT16;
  case popart::cap::DataType::INT32:
    return popart::DataType::INT32;
  case popart::cap::DataType::INT64:
    return popart::DataType::INT64;
  case popart::cap::DataType::UINT32:
    return popart::DataType::UINT32;
  case popart::cap::DataType::UINT64:
    return popart::DataType::UINT64;
  case popart::cap::DataType::BOOL:
    return popart::DataType::BOOL;
  case popart::cap::DataType::FLOAT:
    return popart::DataType::FLOAT;
  case popart::cap::DataType::FLOAT16:
    return popart::DataType::FLOAT16;
  case popart::cap::DataType::BFLOAT16:
    return popart::DataType::BFLOAT16;
  case popart::cap::DataType::DOUBLE:
    return popart::DataType::DOUBLE;
  case popart::cap::DataType::COMPLEX64:
    return popart::DataType::COMPLEX64;
  case popart::cap::DataType::COMPLEX128:
    return popart::DataType::COMPLEX128;
  case popart::cap::DataType::STRING:
    return popart::DataType::STRING;
  case popart::cap::DataType::UNDEFINED:
    return popart::DataType::UNDEFINED;
  }

  std::stringstream errorStream;
  errorStream << "Invalid DataType " << type;
  throw error(errorStream.str());
}

popart::cap::CommGroupType toCapnpCommGroupType(popart::CommGroupType type) {
  switch (type) {
  case popart::CommGroupType::All:
    return popart::cap::CommGroupType::ALL;
  case popart::CommGroupType::Consecutive:
    return popart::cap::CommGroupType::CONSECUTIVE;
  case popart::CommGroupType::Orthogonal:
    return popart::cap::CommGroupType::ORTHOGONAL;
  case popart::CommGroupType::None:
    return popart::cap::CommGroupType::NONE;
  default:
    std::stringstream errorStream;
    errorStream << "Invalid CommGroupType " << type;
    throw error(errorStream.str());
  }
}

popart::CommGroupType toPopartCommGroupType(popart::cap::CommGroupType type) {
  switch (type) {
  case popart::cap::CommGroupType::ALL:
    return popart::CommGroupType::All;
  case popart::cap::CommGroupType::CONSECUTIVE:
    return popart::CommGroupType::Consecutive;
  case popart::cap::CommGroupType::ORTHOGONAL:
    return popart::CommGroupType::Orthogonal;
  case popart::cap::CommGroupType::NONE:
    return popart::CommGroupType::None;
  case popart::cap::CommGroupType::N:
  default:
    std::stringstream errorStream;
    errorStream << "Invalid CommGroupType " << type;
    throw error(errorStream.str());
  }
}

popart::cap::VariableRetrievalMode
toCapnpVariableRetrievalMode(popart::VariableRetrievalMode mode) {
  switch (mode) {
  case popart::VariableRetrievalMode::OnePerGroup:
    return popart::cap::VariableRetrievalMode::ONE_PER_GROUP;
  case popart::VariableRetrievalMode::AllReduceReplicas:
    return popart::cap::VariableRetrievalMode::ALL_REDUCE_REPLICAS;
  case popart::VariableRetrievalMode::AllReplicas:
    return popart::cap::VariableRetrievalMode::ALL_REPLICAS;
  default:
    std::stringstream errorStream;
    errorStream << "Invalid VariableRetrievalMode " << mode;
    throw error(errorStream.str());
  }
}

popart::VariableRetrievalMode
toPopartVariableRetrievalMode(popart::cap::VariableRetrievalMode mode) {
  switch (mode) {
  case popart::cap::VariableRetrievalMode::ONE_PER_GROUP:
    return popart::VariableRetrievalMode::OnePerGroup;
  case popart::cap::VariableRetrievalMode::ALL_REDUCE_REPLICAS:
    return popart::VariableRetrievalMode::AllReduceReplicas;
  case popart::cap::VariableRetrievalMode::ALL_REPLICAS:
    return popart::VariableRetrievalMode::AllReplicas;
  default:
    std::stringstream errorStream;
    errorStream << "Invalid VariableRetrievalMode " << mode;
    throw error(errorStream.str());
  }
}

void serializeTensor(const popart::Tensor *tensor,
                     popart::cap::Tensor::Builder &tensorBuilder,
                     bool serializeTensorData) {
  tensorBuilder.setId(tensor->id);
  tensorBuilder.setTensorType(toCapnpTensorType(tensor->tensorType()));
  auto tensorInfoBuilder   = tensorBuilder.initTensorInfo();
  auto dataTypeInfoBuilder = tensorInfoBuilder.initDataTypeInfo();
  dataTypeInfoBuilder.setDataType(
      toCapnpDataType(tensor->info.getDataTypeInfo()->type()));
  dataTypeInfoBuilder.setNbytes(tensor->info.nbytes());
  dataTypeInfoBuilder.setIsFixedPoint(
      tensor->info.getDataTypeInfo()->isFixedPoint());
  dataTypeInfoBuilder.setName(tensor->info.getDataTypeInfo()->name());
  dataTypeInfoBuilder.setLCaseName(tensor->info.getDataTypeInfo()->lcasename());
  auto shapeBuilder = tensorInfoBuilder.initShape(tensor->info.shape().size());
  for (int j = 0; j < tensor->info.shape().size(); ++j) {
    shapeBuilder.set(j, tensor->info.shape()[j]);
  }

  const auto &locationInfo = tensor->tensorLocationInfo;
  auto locationInfoBuilder = tensorBuilder.initTensorLocationInfo();
  locationInfoBuilder.setRemote(locationInfo.isRemote());
  locationInfoBuilder.setSharded(locationInfo.isSharded());
  auto remoteBufferInfoBuilder = locationInfoBuilder.initRemoteBufferInfo();

  const auto &remoteBufferInfo = locationInfo.getRemoteBufferInfo();
  remoteBufferInfoBuilder.setId(remoteBufferInfo.first);
  remoteBufferInfoBuilder.setIndex(remoteBufferInfo.second);

  if (serializeTensorData) {
    const auto ptr =
        reinterpret_cast<const kj::byte *>(tensor->tensorData()->data());
    auto reader = capnp::Data::Reader(ptr, tensor->tensorData()->size());
    tensorBuilder.setTensorData(reader);
  }

  const auto &variableSettings = tensor->getVariableSettings();
  auto variableSettingsBuilder = tensorBuilder.initVariableSettings();
  variableSettingsBuilder.setRetrievalMode(
      toCapnpVariableRetrievalMode(variableSettings.getRetrievalMode()));

  const auto &sharedVariableDomain = variableSettings.getSharedVariableDomain();
  auto sharedVariableDomainBuilder =
      variableSettingsBuilder.initSharedVariableDomain();
  sharedVariableDomainBuilder.setType(
      toCapnpCommGroupType(sharedVariableDomain.type));
  sharedVariableDomainBuilder.setReplicaGroupSize(
      sharedVariableDomain.replicaGroupSize);
}

std::unique_ptr<popart::Tensor>
deserializeTensor(popart::Ir &ir,
                  const popart::cap::Tensor::Reader &capnpTensor,
                  bool deserializeData) {
  auto gid = popart::GraphId("");
  popart::Graph dummyGraph(ir, gid);
  std::string id        = capnpTensor.getId();
  auto popartTensorType = toPopartTensorType(capnpTensor.getTensorType());

  auto capnpVariableSettings = capnpTensor.getVariableSettings();
  auto capnpSharedVariableDomain =
      capnpVariableSettings.getSharedVariableDomain();
  VariableSettings varSettings(
      CommGroup(toPopartCommGroupType(capnpSharedVariableDomain.getType()),
                capnpSharedVariableDomain.getReplicaGroupSize()),
      toPopartVariableRetrievalMode(capnpVariableSettings.getRetrievalMode()));

  auto tensor = std::make_unique<popart::Tensor>(
      id, popartTensorType, varSettings, dummyGraph);

  auto capnpTensorInfo      = capnpTensor.getTensorInfo();
  auto capnpDataTypeInfo    = capnpTensorInfo.getDataTypeInfo();
  popart::DataType dataType = toPopartDataType(capnpDataTypeInfo.getDataType());
  auto shapeReader          = capnpTensorInfo.getShape();
  std::vector<int64_t> shape;
  for (const auto s : shapeReader) {
    shape.push_back(s);
  }

  tensor->info = popart::TensorInfo(dataType, shape);

  auto capnpTensorLocationInfo = capnpTensor.getTensorLocationInfo();
  tensor->tensorLocationInfo.setSharded(capnpTensorLocationInfo.getSharded());
  tensor->tensorLocationInfo.setRemote(capnpTensorLocationInfo.getRemote());
  tensor->tensorLocationInfo.setRemoteBufferInfo(
      capnpTensorLocationInfo.getRemoteBufferInfo().getId(),
      capnpTensorLocationInfo.getRemoteBufferInfo().getIndex());

  if (deserializeData) {
    // For Onnx-Ir Models, the tensor data of weights is only stored in the
    // ONNX models. For non-Onnx-Ir Models and every other kind of Variable,
    // it is stored in the capnpTensor.
    if (ir.hasOnnxModel() && popartTensorType == popart::TensorType::Variable &&
        popart::onnxutil::isInitializer(ir.getModel(), id)) {

      const auto &tensorProto =
          popart::onnxutil::getTensorProto(ir.getModel(), id);
      auto constData = popart::onnxutil::getConstData(tensorProto);
      if (constData.data == nullptr) {
        throw error("Data for Tensor {} is null", id);
      }

      tensor->setTensorData(constData.info, constData.data);
    } else if (capnpTensor.hasTensorData()) {
      auto tensorDataReader = capnpTensor.getTensorData();
      const void *src       = tensorDataReader.begin();
      tensor->setTensorData(src, tensorDataReader.size());
    }
  }

  return tensor;
}

void serializePopartExecutable(std::ostream &out,
                               const popart::popx::Executablex &executable) {

  ::capnp::MallocMessageBuilder message;
  auto executablexBuilder = message.initRoot<popart::popx::cap::Executablex>();
  auto irLoweringBuilder  = executablexBuilder.initIrLowering();

  auto &ir_lowering = executable.lowering();
  auto &ir          = ir_lowering.ir();
  auto irBuilder    = irLoweringBuilder.initIr();

  irBuilder.setRequiresRandomSeed(ir.getRequiresRandomSeed());

  irBuilder.setExecutionMode(ir.getExecutionMode() ==
                                     Ir::ExecutionMode::Inference
                                 ? popart::cap::Ir::ExecutionMode::INFERENCE
                                 : popart::cap::Ir::ExecutionMode::TRAINING);
  {
    const auto &additionalModelProtoTensors =
        ir.getAdditionalModelProtoTensors();
    auto protoTensorsBuilder = irBuilder.initAdditionalModelProtoTensors(
        additionalModelProtoTensors.size());

    int i = 0;
    for (auto *tensor : additionalModelProtoTensors) {
      protoTensorsBuilder.set(i, tensor->id);
      ++i;
    }
  }

  {
    auto linearlyCreatedInputTensors =
        ir_lowering.getLinearlyCreatedInputTensors();
    auto linearlyCreatedInputTensorsBuilder =
        irLoweringBuilder.initLinearlyCreatedInputTensors(
            linearlyCreatedInputTensors.size());
    int i = 0;
    for (const auto &tid : linearlyCreatedInputTensors) {
      linearlyCreatedInputTensorsBuilder.set(i, tid);
      ++i;
    }
  }

  {
    auto efficientlyCreatedInputTensors =
        ir_lowering.getEfficientlyCreatedInputTensors();
    auto efficientlyCreatedInputTensorsBuilder =
        irLoweringBuilder.initEfficientlyCreatedInputTensors(
            efficientlyCreatedInputTensors.size());
    int i = 0;
    for (const auto &tid : efficientlyCreatedInputTensors) {
      efficientlyCreatedInputTensorsBuilder.set(i, tid);
      ++i;
    }
  }

  {
    auto cycleCountIds = ir_lowering.getCycleCountIds();
    auto cycleCountIdsBuilder =
        irLoweringBuilder.initCycleCountIds(cycleCountIds.size());
    int i = 0;
    for (const auto &tid : cycleCountIds) {
      cycleCountIdsBuilder.set(i, tid);
      ++i;
    }
  }

  {
    // Store the handle (string) and program index (integer) of all custom
    // programs
    auto programHandleIndexMap = ir_lowering.getProgramHandleIndexMap();
    auto programHandleIndicesBuilder =
        irLoweringBuilder.initProgramHandleIndices();
    auto idPairsBuilder =
        programHandleIndicesBuilder.initIdPairs(programHandleIndexMap.size());
    int i = 0;
    for (const auto &programHandleAndIndex : programHandleIndexMap) {
      idPairsBuilder[i].setIndex(programHandleAndIndex.second);
      idPairsBuilder[i].setHandleId(programHandleAndIndex.first);
      ++i;
    }
  }

  {
    auto variableTensors   = ir.getTensorIds(TensorType::Variable);
    auto anchorTensors     = ir.getRootAnchors();
    auto optimizerTensors  = ir.optimizerTensors();
    auto dataStreamTensors = ir.dataStreamTensors();

    size_t numTensorsToSerialize =
        variableTensors.size() + anchorTensors.size() +
        optimizerTensors.size() + dataStreamTensors.size();

    if (ir.getRequiresRandomSeed()) {
      ++numTensorsToSerialize;
    }

    auto tensors = executablexBuilder.initTensors(numTensorsToSerialize);

    size_t i = 0;

    for (auto &id : variableTensors) {
      Tensor *tensor = ir.getTensor(id);
      if (!tensor->hasProducer()) {
        auto tensorBuilder = tensors[i];

        // For Onnx-Ir models, we don't store the tensorData
        // for the variable tensors with initializers since
        // they will be loaded from the onnx file.
        // For Ir models, and others, the tensor data is never serialised
        // as we can use the TensorData in the provided Ir.
        bool serializeTensorData = false;
        if (ir.hasOnnxModel()) {
          bool isInitializer =
              popart::onnxutil::isInitializer(ir.getModel(), id);
          serializeTensorData =
              !isInitializer || tensor->isOptimizerStateTensor();
        }

        serializeTensor(tensor, tensorBuilder, serializeTensorData);
        ++i;
      }
    }

    for (auto &id : anchorTensors) {
      Tensor *tensor     = ir.getTensor(id);
      auto tensorBuilder = tensors[i];
      serializeTensor(tensor, tensorBuilder, false);
      ++i;
    }

    for (auto *tensor : ir.optimizerTensors()) {
      auto tensorBuilder = tensors[i];
      serializeTensor(tensor, tensorBuilder);
      ++i;
    }

    for (auto *tensor : ir.dataStreamTensors()) {
      auto tensorBuilder = tensors[i];
      serializeTensor(tensor, tensorBuilder, false);
      ++i;
    }

    if (ir.getRequiresRandomSeed()) {
      const Tensor *seedTensor = executable.getSeedTensor();
      auto tensorBuilder       = tensors[i];
      serializeTensor(seedTensor, tensorBuilder, true);
      ++i;
    }
  }

  {
    const auto &collectiveBalancedReorderIds =
        ir_lowering.getReplicatedTensorShardingBundle()
            .getCollectiveReorderIds();
    auto hostRearrangementIdsBuilder =
        executablexBuilder.initCollectiveBalancedHostRearrangementIds();
    auto rearrangementIdsBuilder = hostRearrangementIdsBuilder.initIdPairs(
        collectiveBalancedReorderIds.size());

    int i = 0;
    for (const auto &kv : collectiveBalancedReorderIds) {
      rearrangementIdsBuilder[i].setId(kv.first);
      rearrangementIdsBuilder[i].setCbrId(kv.second);
      ++i;
    }
  }

  {
    const auto &collectiveBalancedReorders =
        ir_lowering.getReplicatedTensorShardingBundle().getCollectiveReorders();

    auto hostRearrangementsBuilder =
        executablexBuilder.initCollectiveBalancedHostRearrangements();
    auto rearrangementsBuilder = hostRearrangementsBuilder.initRearrangements(
        collectiveBalancedReorders.size());

    int i = 0;
    for (const auto &kv : collectiveBalancedReorders) {
      rearrangementsBuilder[i].setCbrId(kv.first);

      const auto &hostRearrangement = kv.second->getHostRearrangement();
      auto rearrangementBuilder = rearrangementsBuilder[i].initRearrangement();
      rearrangementBuilder.setReplicationFactor(
          hostRearrangement.getReplicationFactor());

      rearrangementBuilder.setTotalElementsPerReplica(
          hostRearrangement.getTotalElementsPerReplica());

      const auto &gatheredToRefSlices =
          hostRearrangement.getGatheredToRefSlices();
      auto gatheredToRefSlicesBuilder =
          rearrangementBuilder.initGatheredToRefSlices(
              gatheredToRefSlices.size());
      int j = 0;
      for (const auto &s : gatheredToRefSlices) {
        gatheredToRefSlicesBuilder[j].setBegin(s.begin());
        gatheredToRefSlicesBuilder[j].setEnd(s.end());
        ++j;
      }

      ++i;
    }
  }

  kj::std::StdOutputStream sos(out);
  capnp::writeMessage(sos, message);
}

std::unique_ptr<popart::popx::Executablex>
deserializePopartExecutable(std::istream &in,
                            popart::Ir &ir,
                            popart::popx::IrLowering &lowering) {
  kj::std::StdInputStream sis(in);

  capnp::ReaderOptions opts;
  // Increase default size from 64 MB to handle larger models.
  // Note: traversalLimitsInWords is a security check for when Capnp is used as
  // a network communication protocol. It doesn't affect the memory consumption
  // or performance of the library.
  opts.traversalLimitInWords = kj::maxValue;
  capnp::InputStreamMessageReader message(sis, opts);

  auto executablexReader = message.getRoot<popart::popx::cap::Executablex>();
  auto irLoweringReader  = executablexReader.getIrLowering();

  auto irReader = irLoweringReader.getIr();
  if (irReader.getRequiresRandomSeed()) {
    ir.setRequiresRandomSeed();
  }
  {
    auto executionMode = irReader.getExecutionMode();
    if (executionMode == popart::cap::Ir::ExecutionMode::INFERENCE) {
      ir.setExecutionMode(popart::Ir::ExecutionMode::Inference);
    } else {
      ir.setExecutionMode(popart::Ir::ExecutionMode::Training);
    }
  }

  {
    auto linearlyCreatedInputTensors =
        irLoweringReader.getLinearlyCreatedInputTensors();
    std::set<TensorId> linearlyCreatedInputTensors_;
    for (const auto t : linearlyCreatedInputTensors) {
      linearlyCreatedInputTensors_.insert(t);
    }
    lowering.setLinearlyCreatedInputTensors(linearlyCreatedInputTensors_);
  }
  {
    auto efficientlyCreatedInputTensors =
        irLoweringReader.getEfficientlyCreatedInputTensors();
    std::set<TensorId> efficientlyCreatedInputTensors_;
    for (const auto t : efficientlyCreatedInputTensors) {
      efficientlyCreatedInputTensors_.insert(t);
    }
    lowering.setEfficientlyCreatedInputTensors(efficientlyCreatedInputTensors_);
  }
  {
    auto cycleCountIds = irLoweringReader.getCycleCountIds();
    std::vector<TensorId> cycleCountIds_;
    cycleCountIds_.reserve(cycleCountIds.size());
    for (const auto t : cycleCountIds) {
      cycleCountIds_.push_back(t);
    }
    lowering.setCycleCountIds(cycleCountIds_);
  }

  {
    // Restore the handle (string) and program index (integer) of all custom
    // programs
    auto programHandleIndices =
        irLoweringReader.getProgramHandleIndices().getIdPairs();
    std::map<std::string, unsigned> programHandleIndexMap;

    for (const auto &programHandleIndex : programHandleIndices) {
      programHandleIndexMap[programHandleIndex.getHandleId()] =
          programHandleIndex.getIndex();
    }

    lowering.setProgramHandleIndexMap(programHandleIndexMap);
  }

  std::unordered_map<TensorId, std::unique_ptr<popart::Tensor>>
      deserializedTensors;
  {
    auto tensors = executablexReader.getTensors();
    deserializedTensors.reserve(tensors.size());

    for (const auto capnpTensor : tensors) {
      auto tensor                     = deserializeTensor(ir, capnpTensor);
      deserializedTensors[tensor->id] = std::move(tensor);
    }
  }
  {
    // It is unsafe to call 'addAdditionalModelProtoTensors' twice on the Ir.
    // Only call on the passed-by-reference Ir if it is safe to do so.
    if (ir.additionalModelProtoTensorsHaveBeenAdded()) {
      // Check that the Ir we are modifying has expected
      // additionalModelProtoTensors
      std::set<TensorId> irAdditionalIds;
      for (const Tensor *tensor : ir.getAdditionalModelProtoTensors()) {
        irAdditionalIds.insert(tensor->id);
      }
      for (const TensorId id : irReader.getAdditionalModelProtoTensors()) {
        if (!ir.tensorExistsInInitialisers(id) &&
            irAdditionalIds.find(id) == irAdditionalIds.end()) {
          throw error("deserializeExecutable : Deserialization failed. Ir "
                      "passed by reference is already prepared, but tensor "
                      "with TensorId {} in the deserialized executable exists "
                      "in neither its 'additionalModelProtoTensors' nor its "
                      "model proto's initializers.",
                      id);
        }
      }
    } else {
      for (const TensorId id : irReader.getAdditionalModelProtoTensors()) {
        auto *tensor = deserializedTensors[id].get();
        ir.addAdditionalModelProtoTensor(tensor);
      }
      ir.addAdditionalModelProtoTensors();
    }
  }

  std::map<TensorId, CollectiveBalancedReorderId> cbrHostRearrangementIds;
  {
    auto collectiveBalancedHostRearrangementIdsReader =
        executablexReader.getCollectiveBalancedHostRearrangementIds();
    auto idPairsReader =
        collectiveBalancedHostRearrangementIdsReader.getIdPairs();

    for (const auto cbr : idPairsReader) {
      TensorId id                       = cbr.getId();
      CollectiveBalancedReorderId cbrId = cbr.getCbrId();

      cbrHostRearrangementIds[id] = cbrId;
    }
  }

  std::map<CollectiveBalancedReorderId,
           gcl::CollectiveBalancedHostRearrangement>
      cbrHostRearrangements;
  {
    auto collectiveBalancedHostRearrangementsReader =
        executablexReader.getCollectiveBalancedHostRearrangements();
    auto rearrangementsReader =
        collectiveBalancedHostRearrangementsReader.getRearrangements();

    for (const auto cbr : rearrangementsReader) {
      CollectiveBalancedReorderId cbrId = cbr.getCbrId();
      auto rearrangementReader          = cbr.getRearrangement();

      gcl::CollectiveBalancedHostRearrangement cbhr;
      cbhr.setReplicationFactor(rearrangementReader.getReplicationFactor());
      cbhr.setTotalElementsPerReplica(
          rearrangementReader.getTotalElementsPerReplica());

      {
        auto gatheredToRefSlicesReader =
            rearrangementReader.getGatheredToRefSlices();
        std::vector<poplar::Interval> slices;
        slices.reserve(gatheredToRefSlicesReader.size());
        for (const auto s : gatheredToRefSlicesReader) {
          slices.push_back(poplar::Interval(s.getBegin(), s.getEnd()));
        }
        cbhr.setGatheredToRefSlices(std::move(slices));
      }

      cbrHostRearrangements[cbrId] = cbhr;
    }
  }

  auto exe = popart::popx::Executablex::createFromStream(
      lowering,
      std::move(deserializedTensors),
      std::move(cbrHostRearrangementIds),
      std::move(cbrHostRearrangements));

  return exe;
}

} // namespace serialization
} // namespace popx
} // namespace popart
