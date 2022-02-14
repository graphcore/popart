// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cstdlib>

#include <onnxutil.hpp>
#include <popart/graph.hpp>
#include <popart/intervals.hpp>
#include <popart/ir.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/scheduler.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

#include <popart/popx/executablex.hpp>
#include <popart/popx/executablexserialization.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

#include <popart/vendored/optional.hpp>

#include <popef/Reader.hpp>
#include <popef/Writer.hpp>

#include <gcl/CollectiveBalancedReorder.hpp>

#include <capnp/compat/json.h>
#include <capnp/message.h>
#include <capnp/serialize.h>

#include <kj/std/iostream.h>

#include <popart/capnp/Executablex.capnp.h>
#include <popart/capnp/Ir.capnp.h>
#include <popart/capnp/IrLowering.capnp.h>

namespace popart {
namespace popx {
namespace serialization {
namespace {

const std::string popartOpaqueName("popart");

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

popef::DataType toPopefDataType(popart::DataType type) {
  switch (type) {
  case popart::DataType::BOOL:
    return popef::DataType::BOOL;
  case popart::DataType::UINT8:
    return popef::DataType::U8;
  case popart::DataType::INT8:
    return popef::DataType::S8;
  case popart::DataType::UINT16:
    return popef::DataType::U16;
  case popart::DataType::INT16:
    return popef::DataType::S16;
  case popart::DataType::INT32:
    return popef::DataType::S32;
  case popart::DataType::UINT32:
    return popef::DataType::U32;
  case popart::DataType::INT64:
    return popef::DataType::S64;
  case popart::DataType::UINT64:
    return popef::DataType::U64;
  case popart::DataType::FLOAT8:
    return popef::DataType::F8;
  case popart::DataType::FLOAT16:
    return popef::DataType::F16;
  case popart::DataType::FLOAT:
    return popef::DataType::F32;
  case popart::DataType::DOUBLE:
    return popef::DataType::F64;
  default:
    std::stringstream errorStream;
    errorStream << "There is no popef mapping for popart::DataType " << type;
    throw error(errorStream.str());
  }
}

void serializeTensor(const popart::Tensor *tensor,
                     popart::cap::Tensor::Builder &tensorBuilder,
                     bool serializeTensorData = true) {
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
    auto reader = capnp::Data::Reader(ptr, tensor->info.nbytes());
    tensorBuilder.setTensorData(reader);
  }
}

std::unique_ptr<popart::Tensor>
deserializeTensor(popart::Ir &ir,
                  const popart::cap::Tensor::Reader &capnpTensor,
                  bool deserializeData = true) {
  auto gid = popart::GraphId("");
  popart::Graph dummyGraph(ir, gid);
  std::string id        = capnpTensor.getId();
  auto popartTensorType = toPopartTensorType(capnpTensor.getTensorType());
  auto tensor =
      std::make_unique<popart::Tensor>(id, popartTensorType, dummyGraph);

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

      if (constData.info != tensor->info) {
        throw error("TensorInfo mismatch for {}, expected {}, got {}",
                    id,
                    tensor->info,
                    constData.info);
      }

      tensor->setTensorData(tensor->info, constData.data);
    } else if (capnpTensor.hasTensorData()) {
      auto tensorDataReader = capnpTensor.getTensorData();
      const void *src       = tensorDataReader.begin();
      tensor->setTensorData(tensor->info, src);
    }
  }

  return tensor;
}

int getIpuVersion(const IrLowering &ir_lowering) {
  const std::string targetArchStr = "ipu";
  const std::string archVersionStr =
      ir_lowering.getDeviceInfo()->getTarget().getTargetArchString();
  const size_t pos           = archVersionStr.find(targetArchStr);
  const bool ipuIsUsed       = pos != std::string::npos;
  const size_t ipuVersionPos = ipuIsUsed ? pos + targetArchStr.size() : pos;

  try {
    const std::string ipuVersionStr = archVersionStr.substr(ipuVersionPos);
    return std::atoi(ipuVersionStr.c_str());
  } catch (...) {
    throw error("Cannot get ipu version from target architecture string " +
                archVersionStr +
                ". Expected the target architecture string to contain a "
                "substring of the form ^ipu[digits]$");
  }
}

bool isPOD(const IrLowering &ir_lowering) {
  const std::string targetSystemStr = "POD";
  const std::string systemVersionStr =
      ir_lowering.getDeviceInfo()->getTarget().getTargetSystemString();
  return systemVersionStr.find(targetSystemStr) != std::string::npos;
}

std::vector<popef::Option>
convertOptionFlagsToOptions(const poplar::OptionFlags &optFlags) {
  std::vector<popef::Option> opts;
  std::transform(optFlags.begin(),
                 optFlags.end(),
                 std::back_inserter(opts),
                 [](const auto &optFlag) {
                   return popef::Option(optFlag.first, optFlag.second);
                 });
  return opts;
}

popef::TensorInfo createTensorInfo(const popart::Tensor &tensor) {
  popef::TensorInfo tensorInfo;
  tensorInfo.setDataType(
      toPopefDataType(tensor.info.getDataTypeInfo()->type()));
  tensorInfo.setShape(tensor.info.shape());
  return tensorInfo;
}

const popef::Anchor &putAnchorToMetadata(const popart::Tensor &tensor,
                                         const bool isPerReplica,
                                         const popef::TensorType tensorType,
                                         nonstd::optional<bool> isAnchorStream,
                                         popef::Metadata &metadata) {
  popef::Anchor anchor;
  anchor.setName(tensor.id);
  anchor.setTensorInfo(createTensorInfo(tensor));
  anchor.setIsPerReplica(isPerReplica);
  anchor.setType(tensorType);

  if (tensorType == popef::TensorType::INPUT) {
    anchor.setHandle(IrLowering::h2dId(tensor.id));
  } else if (tensorType == popef::TensorType::OUTPUT) {
    anchor.setHandle(IrLowering::d2hId(tensor.id, *isAnchorStream));
  }

  metadata.anchors().push_back(anchor);
  return metadata.anchors().back();
}

void serializePopefTensor(const popart::Tensor &tensor,
                          const popef::TensorInfo &tensorInfo,
                          popef::Writer &writer) {
  popef::TensorDataInfo tensorDataInfo;
  tensorDataInfo.setName(tensor.id);
  tensorDataInfo.setTensorInfo(tensorInfo);

  std::shared_ptr<popef::BlobWriter> tensorWriter =
      writer.createTensorData(tensorDataInfo);
  const char *dataPtr =
      reinterpret_cast<const char *>(tensor.tensorData()->data());
  tensorWriter->stream.write(dataPtr, tensor.info.nbytes());
}

void serializePopefAnchors(const popart::popx::Executablex &popartMetadata,
                           popef::Metadata &metadata,
                           popef::Writer &writer) {
  for (auto *tensor : popartMetadata.getWeightTensors()) {
    if (tensor->hasProducer()) {
      throw error("Weights are tensors of variable type that do not"
                  "have producers.");
    }

    static constexpr bool isPerReplica   = false;
    static constexpr bool isAnchorStream = false;
    const popef::Anchor *anchor_ptr      = nullptr;

    auto &ir = popartMetadata.lowering().ir();
    if (!ir.streamingIsDisabledForTensor(tensor)) {
      // Tensor has to be handled as an input and output stream.
      // Add the input anchor
      putAnchorToMetadata(*tensor,
                          isPerReplica,
                          popef::TensorType::INPUT,
                          nonstd::nullopt,
                          metadata);
      // Add the output anchor
      anchor_ptr = &putAnchorToMetadata(*tensor,
                                        isPerReplica,
                                        popef::TensorType::OUTPUT,
                                        isAnchorStream,
                                        metadata);
    } else {
      anchor_ptr = &putAnchorToMetadata(*tensor,
                                        isPerReplica,
                                        popef::TensorType::UNKNOWN,
                                        nonstd::nullopt,
                                        metadata);
    }
    serializePopefTensor(*tensor, anchor_ptr->tensorInfo(), writer);
  }

  for (Tensor *tensor : popartMetadata.getAnchorTensors()) {
    static constexpr bool isPerReplica   = true;
    static constexpr bool isAnchorStream = true;
    putAnchorToMetadata(*tensor,
                        isPerReplica,
                        popef::TensorType::OUTPUT,
                        isAnchorStream,
                        metadata);
  }

  if (popartMetadata.getSeedTensor() != nullptr) {
    static constexpr bool isPerReplica = false;
    const popef::Anchor &anchor =
        putAnchorToMetadata(*popartMetadata.getSeedTensor(),
                            isPerReplica,
                            popef::TensorType::INPUT,
                            nonstd::nullopt,
                            metadata);
    metadata.setSeedHandle(anchor.handle());
    serializePopefTensor(
        *popartMetadata.getSeedTensor(), anchor.tensorInfo(), writer);
  }

  for (auto *tensor : popartMetadata.getOptimizerTensors()) {
    static constexpr bool isPerReplica = false;
    const popef::Anchor &anchor        = putAnchorToMetadata(*tensor,
                                                      isPerReplica,
                                                      popef::TensorType::INPUT,
                                                      nonstd::nullopt,
                                                      metadata);
    serializePopefTensor(*tensor, anchor.tensorInfo(), writer);
  }

  for (Tensor *tensor : popartMetadata.getDataStreamTensors()) {
    static constexpr bool isPerReplica = true;
    putAnchorToMetadata(*tensor,
                        isPerReplica,
                        popef::TensorType::INPUT,
                        nonstd::nullopt,
                        metadata);
  }
}

void serializePopefMetadata(const popart::popx::Executablex &popartMetadata,
                            const std::string &programHash,
                            popef::Writer &writer) {
  auto &ir_lowering = popartMetadata.lowering();
  auto &ir          = ir_lowering.ir();
  auto &opts        = ir.getSessionOptions();

  const bool isInference =
      ir.getExecutionMode() == Ir::ExecutionMode::Inference;
  const int64_t numProcs =
      opts.enableDistributedReplicatedGraphs ? opts.globalReplicationFactor : 1;

  popef::Metadata metadata;
  metadata.setReplicationFactor(ir_lowering.getReplicationFactor());
  metadata.setNumIpus(ir_lowering.getDeviceInfo()->getNumIpus());
  metadata.setIpuVersion(getIpuVersion(ir_lowering));
  metadata.setExecutable(programHash);
  metadata.setIsPOD(isPOD(ir_lowering));
  metadata.setNumProcesses(numProcs);
  metadata.setIsInference(isInference);
  metadata.setEngineOptions(
      convertOptionFlagsToOptions(ir_lowering.engineOptions));
  metadata.setDeviceOptions(convertOptionFlagsToOptions(
      ir_lowering.getDeviceInfo()->getOptionFlags()));

  serializePopefAnchors(popartMetadata, metadata, writer);

  popef::ProgramFlow flow;
  flow.setLoad({PopPrograms::ProgramIndex::WeightsFromHost,
                PopPrograms::ProgramIndex::OptimizerFromHost,
                PopPrograms::ProgramIndex::RandomSeedFromHost,
                PopPrograms::ProgramIndex::RngStateFromHost});
  flow.setMain({PopPrograms::ProgramIndex::Program});
  flow.setSave({PopPrograms::ProgramIndex::WeightsToHost,
                PopPrograms::ProgramIndex::RandomSeedToHost,
                PopPrograms::ProgramIndex::RngStateToHost,
                PopPrograms::ProgramIndex::CycleCountTensorToHost});
  metadata.setProgramFlow(flow);

  writer.write(metadata);
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
        // For Ir models, and others, the tensor data is always serialized
        bool serializeTensorData = true;
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
      TensorId seedId    = GetRandomSeedOp::getStreamedSeedTensorId();
      Tensor *seedTensor = ir.getTensor(seedId);
      auto tensorBuilder = tensors[i];
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
          ir_lowering.getReplicationFactor());

      rearrangementBuilder.setTotalElementsPerReplica(
          hostRearrangement.totalElementsPerReplica);

      const auto &gatheredToRefSlices = hostRearrangement.gatheredToRefSlices;
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
deserializePopartMetadata(std::istream &in,
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
      cbhr.replicationFactor = rearrangementReader.getReplicationFactor();
      cbhr.totalElementsPerReplica =
          rearrangementReader.getTotalElementsPerReplica();

      auto gatheredToRefSlicesReader =
          rearrangementReader.getGatheredToRefSlices();
      cbhr.gatheredToRefSlices.reserve(gatheredToRefSlicesReader.size());
      for (const auto s : gatheredToRefSlicesReader) {
        cbhr.gatheredToRefSlices.push_back(
            poplar::Interval(s.getBegin(), s.getEnd()));
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
} // namespace

void serializeEngineExecutable(std::ostream &out,
                               const poplar::Engine *poplarEngine,
                               const popart::popx::Executablex *executable,
                               size_t hash) {
  const std::string programHash = std::to_string(hash);
  popef::Writer popefWriter(out);

  // Export Popart specific data
  if (executable) {
    std::shared_ptr<popef::BlobWriter> popefOpaque =
        popefWriter.createOpaqueBlob(popartOpaqueName, programHash);
    serializePopartExecutable(popefOpaque->stream, *executable);

    try {
      serializePopefMetadata(*executable, programHash, popefWriter);
    } catch (const std::exception &e) {
      logging::session::warn(
          "Serializing popef metadata ended with failure due to {}. The popef "
          "file cannot be used to run model using model_runtime",
          e.what());
    }
  }

  // Export Poplar engine's executable
  if (poplarEngine) {
    static constexpr bool compress = false;
    std::shared_ptr<popef::BlobWriter> popefExe =
        popefWriter.createExecutable(programHash, compress);
    poplarEngine->serializeExecutable(popefExe->stream);
  }
}

class ReaderImpl {
public:
  template <typename T>
  using optional_ref    = nonstd::optional<std::reference_wrapper<T>>;
  using OpaqueReaderOpt = optional_ref<const popef::OpaqueReader>;
  using ExecReaderOpt   = optional_ref<const popef::ExecutableReader>;
  using MetadataOpt     = optional_ref<const popef::Metadata>;
  using TensorReaderVec =
      std::vector<std::reference_wrapper<const popef::TensorReader>>;
  using OpaqueReaderIt = std::vector<popef::OpaqueReader>::const_iterator;
  using ExecReaderIt   = std::vector<popef::ExecutableReader>::const_iterator;
  using MetadataIt     = std::vector<popef::Metadata>::const_iterator;
  using TensorReaderIt = std::vector<popef::TensorReader>::const_iterator;

  ReaderImpl(std::shared_ptr<std::istream> in)
      : popefReader(setupReader(in)), popartMetadata(findPopartMetadata()),
        poplarExecutable(findPoplarExecutable()),
        popefMetadata(findPopefMetadata()), tensorData(findPopefTensors()),
        hash(getExecutableHash()) {}

  popef::Reader popefReader;
  const OpaqueReaderOpt popartMetadata;
  const ExecReaderOpt poplarExecutable;
  const MetadataOpt popefMetadata;
  const TensorReaderVec tensorData;
  const size_t hash;

private:
  popef::Reader setupReader(std::shared_ptr<std::istream> in) {
    popef::Reader reader;
    reader.parseStream(in);
    return reader;
  }

  OpaqueReaderOpt findPopartMetadata() {
    auto popartOpaqueMatcher = [](const popef::OpaqueReader &opaque) {
      return opaque.name.find(popartOpaqueName) != std::string::npos;
    };

    const std::vector<popef::OpaqueReader> &opaques = popefReader.opaqueBlobs();
    const int numOfMatchedPopartMetadata =
        std::count_if(opaques.begin(), opaques.end(), popartOpaqueMatcher);
    if (numOfMatchedPopartMetadata > 1) {
      throw error("Contains more than one Popart metadata");
    }
    OpaqueReaderIt opaqueIt =
        std::find_if(opaques.begin(), opaques.end(), popartOpaqueMatcher);

    const bool opaqueExists = opaqueIt != opaques.end();
    const OpaqueReaderOpt opaqueReader =
        opaqueExists ? OpaqueReaderOpt(*opaqueIt) : nonstd::nullopt;
    return opaqueReader;
  }

  ExecReaderOpt findPoplarExecutable() {
    const std::vector<popef::ExecutableReader> &execs =
        popefReader.executables();

    ExecReaderIt execIt = execs.end();
    if (popartMetadata.has_value()) {
      auto poplarExecMatcher =
          [this](const popef::ExecutableReader &executable) {
            return executable.name == popartMetadata->get().executable;
          };

      const int numOfMatchedPoplarExecs =
          std::count_if(execs.begin(), execs.end(), poplarExecMatcher);
      if (numOfMatchedPoplarExecs > 1) {
        throw error("The file contains more than one poplar executables "
                    "that matches popart metadata.");
      }

      execIt = std::find_if(execs.begin(), execs.end(), poplarExecMatcher);
    } else {
      if (execs.size() > 1) {
        throw error("The popart metadata associated with poplar "
                    "executable does not exist and the popef file "
                    "contains more than one executable, hence the "
                    "correct one cannot be selected.");
      }
      execIt = execs.begin();
    }

    const bool executableExists = execIt != execs.end();
    ExecReaderOpt execReader =
        executableExists ? ExecReaderOpt(*execIt) : nonstd::nullopt;
    return execReader;
  }

  MetadataOpt findPopefMetadata() {
    auto metadataMatcher = [this](const popef::Metadata &metadata) {
      if (!popartMetadata.has_value())
        return false;
      return metadata.executable() == popartMetadata->get().executable;
    };

    const std::vector<popef::Metadata> &metadataVec = popefReader.metadata();
    const int numOfMatchedPopefMetadata =
        std::count_if(metadataVec.begin(), metadataVec.end(), metadataMatcher);
    if (numOfMatchedPopefMetadata > 1) {
      throw error("Contains more than one Popef metadata");
    }
    MetadataIt metadataIt =
        std::find_if(metadataVec.begin(), metadataVec.end(), metadataMatcher);

    const bool metadataExists = metadataIt != metadataVec.end();
    return metadataExists ? MetadataOpt(*metadataIt) : nonstd::nullopt;
  }

  TensorReaderVec findPopefTensors() {
    auto tensorMatcher = [this](const popef::TensorReader &tensor) {
      if (!popefMetadata.has_value())
        return false;
      auto anchorMatcher = [&tensor](const popef::Anchor &anchor) {
        return anchor.name() == tensor.info.name();
      };
      const std::vector<popef::Anchor> &anchors =
          popefMetadata->get().anchors();
      auto anchorIt =
          std::find_if(anchors.begin(), anchors.end(), anchorMatcher);
      return anchorIt != anchors.end();
    };

    TensorReaderVec tensors;
    const std::vector<popef::TensorReader> &allTensors = popefReader.tensors();
    std::copy_if(allTensors.begin(),
                 allTensors.end(),
                 std::back_inserter(tensors),
                 tensorMatcher);

    return tensors;
  }

  size_t getExecutableHash() const {
    size_t hash = 0;

    if (poplarExecutable.has_value() || popartMetadata.has_value()) {
      const std::string &hashString = poplarExecutable.has_value()
                                          ? poplarExecutable->get().name
                                          : popartMetadata->get().executable;
      std::stringstream ss(hashString);
      ss >> hash;
      if (ss.fail()) {
        throw error("Neither the poplar executable nor the popart metadata "
                    "contains a hash number.");
      }
    }

    return hash;
  }
};

Reader::Reader(std::shared_ptr<std::istream> in)
    : _impl(std::make_unique<ReaderImpl>(in)) {}
Reader::Reader(Reader &&reader) : _impl(std::move(reader._impl)) {}
Reader::~Reader() = default;

size_t Reader::readExecutableHash() const { return _impl->hash; }

bool Reader::containsPoplarExecutable() const {
  return _impl->poplarExecutable.has_value();
}

bool Reader::containsExecutable() const {
  return _impl->popartMetadata.has_value();
}

bool Reader::containsPopefMetadata() {
  return _impl->popefMetadata.has_value();
}

poplar::Executable Reader::deserializePoplarExecutable() const {
  if (!containsPoplarExecutable()) {
    throw error("The file does not contain poplar executable.");
  }

  const popef::ExecutableReader &exeReader = _impl->poplarExecutable->get();
  return poplar::Executable::deserialize(
      exeReader.getStandaloneExecutableStream());
}

std::unique_ptr<popart::popx::Executablex>
Reader::deserializeExecutable(popart::Ir &ir,
                              popart::popx::IrLowering &lowering) const {
  if (!containsExecutable()) {
    throw error("The file does not contain popart metadata.");
  }

  const popef::OpaqueReader &metadataReader = _impl->popartMetadata->get();
  std::unique_ptr<std::istream> opaque_stream(
      metadataReader.getStandaloneDataStream());
  auto popartMetadata = deserializePopartMetadata(*opaque_stream, ir, lowering);

  return popartMetadata;
}

} // namespace serialization
} // namespace popx
} // namespace popart
