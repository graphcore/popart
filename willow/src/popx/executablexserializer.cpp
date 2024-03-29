// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "executablexserializer.hpp"

#include "popart/util/expressionchecking.hpp"
#include <algorithm>
#include <capnp/blob.h>
#include <capnp/list.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <cstdint>
#include <cstdlib>
#include <gcl/CollectiveBalancedReorder.hpp>
#include <iterator>
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

#include "boost/range/adaptor/filtered.hpp"
#include "boost/range/adaptor/transformed.hpp"
#include "boost/range/algorithm/copy.hpp"
#include "popart/capnp/Executablex.capnp.h"
#include "popart/capnp/Ir.capnp.h"
#include "popart/capnp/IrLowering.capnp.h"
#include "popart/commgroup.hpp"
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

namespace {

/**
 * The function finds tensor data blob from vector of blobs that
 * matches passed tensor id.
 *
 * \param tensorDataVec Vector of readable PopEF tensor data blobs.
 *                      They contain the serialized data for popart
 *                      tensors.
 * \param tensorId Tensor name that the function looks for in the
 *                 tensor readers.
 * \return Pointer to the tensor reader if tensor id matches with some
 *         tensor reader, nullptr otherwise.
 */
const popef::TensorReader *
getTensorReader(const std::vector<popef::TensorReader> &tensorDataVec,
                const std::string &tensorId) {
  auto tensorDataIt =
      std::find_if(tensorDataVec.cbegin(),
                   tensorDataVec.cend(),
                   [&tensorId](const popef::TensorReader &tensor) {
                     return tensor.info.name() == tensorId;
                   });
  const popef::TensorReader *tensorReader =
      tensorDataIt != tensorDataVec.cend() ? &(*tensorDataIt) : nullptr;
  return tensorReader;
}

} // namespace

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
  // TODO: T69675 add support for FLOAT8 popef serialisation.
  case popart::DataType::FLOAT8_143:
  case popart::DataType::FLOAT8_152:
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
                     popart::cap::Tensor::Builder &tensorBuilder) {
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

  const auto &variableSettings = tensor->getVariableSettings();
  auto variableSettingsBuilder = tensorBuilder.initVariableSettings();
  variableSettingsBuilder.setUseCommGroup(variableSettings.isUsingCommGroup());
  variableSettingsBuilder.setCommGroupType(
      toCapnpCommGroupType(variableSettings.getCommGroupType()));
  variableSettingsBuilder.setStride(variableSettings.getStride());
  variableSettingsBuilder.setGroupSize(variableSettings.getGroupSize());
  variableSettingsBuilder.setRetrievalMode(
      toCapnpVariableRetrievalMode(variableSettings.getRetrievalMode()));
}

std::unique_ptr<popart::Tensor>
deserializeTensor(popart::Ir &ir,
                  const popart::cap::Tensor::Reader &capnpTensor,
                  const popef::TensorReader *tensorReader) {
  auto gid = popart::GraphId("");
  popart::Graph dummyGraph(ir, gid);
  std::string id        = capnpTensor.getId();
  auto popartTensorType = toPopartTensorType(capnpTensor.getTensorType());

  auto capnpVariableSettings = capnpTensor.getVariableSettings();
  auto useCommGroup          = capnpVariableSettings.getUseCommGroup();
  auto commGroupType =
      toPopartCommGroupType(capnpVariableSettings.getCommGroupType());
  auto stride    = capnpVariableSettings.getStride();
  auto groupSize = capnpVariableSettings.getGroupSize();
  auto retrievalMode =
      toPopartVariableRetrievalMode(capnpVariableSettings.getRetrievalMode());
  VariableSettings varSettings =
      useCommGroup
          ? VariableSettings(CommGroup(commGroupType, groupSize), retrievalMode)
          : VariableSettings(stride, groupSize, retrievalMode);

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

  // For Onnx-Ir Models, the tensor data of weights is stored in the
  // ONNX models so we don't have to deserialize tensor data from PopEF
  // tensor data blob. For non-Onnx-Ir Models and every other kind of Variable,
  // we have to.
  if (ir.hasOnnxModel() && popartTensorType == popart::TensorType::Variable &&
      popart::onnxutil::isInitializer(ir.getModel(), id)) {

    const auto &tensorProto =
        popart::onnxutil::getTensorProto(ir.getModel(), id);
    auto constData = popart::onnxutil::getConstData(tensorProto);
    if (constData.data == nullptr) {
      throw error("Data for Tensor {} is null", id);
    }

    tensor->setTensorDataFromCopyOf(constData.data, constData.info.nbytes());
  } else if (tensorReader) {
    const size_t bufferSize = tensorReader->info.tensorInfo().sizeInBytes();
    std::vector<char> tensorBuffer(bufferSize);
    std::unique_ptr<std::istream> tensorStream(
        tensorReader->getStandaloneDataStream());
    tensorStream->read(tensorBuffer.data(), bufferSize);
    POPART_ASSERT_EQ(tensorBuffer.size(), bufferSize);
    tensor->setTensorDataByEmplaceOf(std::move(tensorBuffer));
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
    auto getTensor     = [&ir](const TensorId &id) { return ir.getTensor(id); };
    auto hasNoProducer = [](const Tensor *tensor) {
      return !tensor->hasProducer();
    };

    std::vector<const Tensor *> tensorsToSerialize;

    const auto anchorTensorIds   = ir.getRootAnchors();
    const auto variableTensorIds = ir.getTensorIds(TensorType::Variable);

    boost::copy(anchorTensorIds | boost::adaptors::transformed(getTensor),
                std::back_inserter(tensorsToSerialize));
    boost::copy(variableTensorIds | boost::adaptors::transformed(getTensor) |
                    boost::adaptors::filtered(hasNoProducer),
                std::back_inserter(tensorsToSerialize));
    boost::copy(ir.optimizerTensors(), std::back_inserter(tensorsToSerialize));
    boost::copy(ir.dataStreamTensors(), std::back_inserter(tensorsToSerialize));
    if (ir.getRequiresRandomSeed()) {
      tensorsToSerialize.push_back(executable.getSeedTensor());
    }

    auto tensors = executablexBuilder.initTensors(tensorsToSerialize.size());

    for (std::size_t i = 0; i < tensors.size(); i++) {
      auto tensorBuilder = tensors[i];
      serializeTensor(tensorsToSerialize[i], tensorBuilder);
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

std::unique_ptr<popart::popx::Executablex> deserializePopartExecutable(
    std::istream &in,
    popart::Ir &ir,
    popart::popx::IrLowering &lowering,
    const std::vector<popef::TensorReader> &tensorDataVec) {
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
      const std::string id = capnpTensor.getId();
      const popef::TensorReader *tensorDataReader =
          getTensorReader(tensorDataVec, id);
      auto tensor = deserializeTensor(ir, capnpTensor, tensorDataReader);
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
