// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <istream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popef/Reader.hpp>
#include <popef/Types.hpp>
#include <popef/Writer.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Executable.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/StringRef.hpp>
#include <poplar/Target.hpp>
#include <popx/executablexserializer.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/popefserializer.hpp>
#include <popart/popx/popprograms.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/popx/devicex.hpp"
#include "popx/rng/rngstatelowering.hpp"

namespace popart {
namespace popx {
namespace serialization {
namespace {

using RngStateBufferType = std::map<uint16_t, std::vector<uint32_t>>;

const std::string popartOpaqueName("popart");

/**
 * \return \c popef::DataType casted from \c popart::DataType.
 */
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

/**
 * \param deviceInfo Represents a target device of the model.
 * \return The ipu version that has been declared by the user for compilation
 *         the model.
 */
int getIpuVersion(const DeviceInfo &deviceInfo) {
  const std::string targetArchStr = "ipu";
  const std::string archVersionStr =
      deviceInfo.getTarget().getTargetArchString();
  const size_t pos           = archVersionStr.find(targetArchStr);
  const bool ipuIsUsed       = pos != std::string::npos;
  const size_t ipuVersionPos = ipuIsUsed ? pos + targetArchStr.size() : pos;

  try {
    const std::string ipuVersionStr = archVersionStr.substr(ipuVersionPos);
    return std::atoi(ipuVersionStr.c_str());
  } catch (...) {
    throw error("Cannot get ipu version from target architecture string {}"
                ". Expected the target architecture string to contain a "
                "substring of the form ^ipu[digits]$",
                archVersionStr);
  }
}

/**
 * \param deviceInfo Represents a target device of the model.
 * \return The information if the target runtime system is POD.
 */
bool isPOD(const DeviceInfo &deviceInfo) {
  const std::string targetSystemStr = "POD";
  const std::string systemVersionStr =
      deviceInfo.getTarget().getTargetSystemString();
  return systemVersionStr.find(targetSystemStr) != std::string::npos;
}

/**
 * \param optFlags A set of option and value
 *        string flags.
 * \return vector of \c popef::Option casted from vector of
 *         \c poplar::OptionFlags.
 */
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

/**
 * \param dt The tensor data type.
 * \param shape The tensor shape.
 * \return Creates \c popef::TensorInfo
 */
popef::TensorInfo createTensorInfo(const popef::DataType dt,
                                   const std::vector<int64_t> &shape) {
  popef::TensorInfo tensorInfo;
  tensorInfo.setDataType(dt);
  tensorInfo.setShape(shape);
  return tensorInfo;
}

/**
 * \param info Object that contains information about tensor.
 * \return \c popef::TensorInfo casted from \c popart::TensorInfo.
 */
popef::TensorInfo createTensorInfo(const popart::TensorInfo &info) {
  popef::TensorInfo tensorInfo;
  const popef::DataType dt = toPopefDataType(info.getDataTypeInfo()->type());
  return createTensorInfo(dt, info.shape());
}

/**
 * Creates \c popef::Anchor and inserts it to anchors vector
 * in \c popef::Metadata.
 *
 * \param name Tensor name.
 * \param handle A string key that connects the callback with the tensor.
 * \param isPerReplica Information if this tensor should have separate
 *                     data for each replica
 * \param tensorType Information if this is input or output tensor.
 * \param TensorInfo Object that contains information about tensor.
 * \param metadata Data needed to run a model using Model Runtime.
 * \return \c popef::Anchor that was inserted to the \c popef::Metadata.
 */
popef::Anchor putAnchorToMetadata(const std::string &name,
                                  const PopStreamId &handle,
                                  const bool isPerReplica,
                                  const popef::TensorType tensorType,
                                  const popef::TensorInfo &tensorInfo,
                                  const std::vector<int64_t> &programs,
                                  popef::Metadata &metadata) {
  popef::Anchor anchor;
  anchor.setName(name);
  anchor.setHandle(handle);
  anchor.setTensorInfo(tensorInfo);
  anchor.setIsPerReplica(isPerReplica);
  anchor.setType(tensorType);
  anchor.setPrograms(programs);

  metadata.anchors().push_back(anchor);
  return metadata.anchors().back();
}

/**
 * Serializes content of the tensor to the popef file as TensorBlob.
 *
 * \param tensor Popart tensor.
 * \param tensorInfo Object that contains information about tensor.
 * \param writer Write popef blobs to a given stream.
 */
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

/**
 * Serializes content of the rng state buffer to the popef file
 * as TensorBlob.
 *
 * \param tensorName The rng state buffer name.
 * \param repFactor Number of replicas.
 * \param dt The element data type of rng state buffer.
 * \param shape The rng state bufffer shape.
 * \param rngBuffer Content of rng state buffer for all replicas.
 * \param writer Write popef blobs to a given stream.
 * \return Info if buffer was serialized.
 */
bool serializeRngStateAsPopefTensor(const std::string &tensorName,
                                    const int repFactor,
                                    const popef::DataType dt,
                                    const popart::Shape &shape,
                                    const RngStateBufferType &rngBuffer,
                                    popef::Writer &writer) {
  if (!rngBuffer.empty()) {
    if (rngBuffer.size() != repFactor) {
      throw error("RNG State buffer does not have data for all replicas. "
                  "Devicex did not allocate data for all replicas.");
    }

    popart::Shape shapeWithReplicas(shape);
    shapeWithReplicas.insert(shapeWithReplicas.begin(), repFactor);

    popef::TensorDataInfo tensorDataInfo;
    tensorDataInfo.setName(tensorName);
    tensorDataInfo.setTensorInfo(createTensorInfo(dt, shapeWithReplicas));
    const int64_t nbytes = tensorDataInfo.tensorInfo().sizeInBytes();
    std::shared_ptr<popef::BlobWriter> tensorWriter =
        writer.createTensorData(tensorDataInfo);

    for (int i = 0; i < repFactor; i++) {
      const auto rngStateIt = rngBuffer.find(i);
      if (rngStateIt == rngBuffer.end()) {
        throw error("RNGState data for replica {} does not exist. Devicex "
                    "should allocate it.",
                    i);
      }
      try {
        const char *dataPtr =
            reinterpret_cast<const char *>(rngStateIt->second.data());
        tensorWriter->stream.write(dataPtr, nbytes / repFactor);
      } catch (...) {
        throw error("RNGState data for replica {} does not have enough data.",
                    i);
      }
    }

    return true;
  }

  return false;
}

/**
 * Serializes weights as popef anchors(metadata about in or out tensors) and its
 * data as TensorBlobs.
 *
 * \param executablex The final executable which contains all the data, metadata
 *                    and configuration parameters necessary to start running
 *                    the program on the device.
 * \param metadata Data needed to run a model using Model Runtime.
 * \param writer Write popef blobs to a given stream.
 */
void serializeWeightsAsPopefAnchors(
    const popart::popx::Executablex &executablex,
    popef::Metadata &metadata,
    popef::Writer &writer) {
  const auto &ir = executablex.lowering().ir();

  for (auto *tensor : executablex.getWeightTensors()) {
    if (tensor->hasProducer()) {
      throw error("Weights are tensors of variable type that do not"
                  "have producers.");
    }

    static constexpr bool isPerReplica   = false;
    static constexpr bool isAnchorStream = false;
    const popef::TensorInfo tensorInfo   = createTensorInfo(tensor->info);

    if (!ir.streamingIsDisabledForTensor(tensor)) {
      // Tensor has to be handled as an input and output stream
      // as there are programs responsible for loading and storing
      // weights.
      const popef::Anchor anchor =
          putAnchorToMetadata(tensor->id,
                              IrLowering::h2dId(tensor->id),
                              isPerReplica,
                              popef::TensorType::INPUT,
                              tensorInfo,
                              {PopPrograms::ProgramIndex::WeightsFromHost},
                              metadata);
      putAnchorToMetadata(tensor->id,
                          IrLowering::d2hId(tensor->id, isAnchorStream),
                          isPerReplica,
                          popef::TensorType::OUTPUT,
                          tensorInfo,
                          {PopPrograms::ProgramIndex::WeightsToHost},
                          metadata);
      serializePopefTensor(*tensor, anchor.tensorInfo(), writer);
    } else {
      const popef::Anchor anchor =
          putAnchorToMetadata(tensor->id,
                              "",
                              isPerReplica,
                              popef::TensorType::UNKNOWN,
                              tensorInfo,
                              {},
                              metadata);
      serializePopefTensor(*tensor, anchor.tensorInfo(), writer);
    }
  }
}

/**
 * Serializes optimizer tensors as popef anchors(metadata about in or out
 * tensor) and its data as TensorBlobs. Note: The optimizer parameter tensors
 * are e.g. learning rate(s), momentum(s), weight decay factor(s), loss scaling.
 *
 * \param executablex The final executable which contains all the data, metadata
 *                    and configuration parameters necessary to start running
 *                    the program on the device.
 * \param metadata Data needed to run a model using Model Runtime.
 * \param writer Write popef blobs to a given stream.
 */
void serializeOptimizersAsPopefAnchors(
    const popart::popx::Executablex &executablex,
    popef::Metadata &metadata,
    popef::Writer &writer) {
  for (auto *tensor : executablex.getOptimizerTensors()) {
    static constexpr bool isPerReplica = false;
    const popef::Anchor anchor =
        putAnchorToMetadata(tensor->id,
                            IrLowering::h2dId(tensor->id),
                            isPerReplica,
                            popef::TensorType::INPUT,
                            createTensorInfo(tensor->info),
                            {PopPrograms::ProgramIndex::OptimizerFromHost},
                            metadata);
    serializePopefTensor(*tensor, anchor.tensorInfo(), writer);
  }
}

/**
 * Serializes data stream(user inputs) tensors as popef anchors(metadata about
 * in or out tensors).
 *
 * \param executablex The final executable which contains all the data, metadata
 *                    and configuration parameters necessary to start running
 *                    the program on the device.
 * \param metadata Data needed to run a model using Model Runtime.
 */
void serializeDataStreamsAsPopefAnchors(
    const popart::popx::Executablex &executablex,
    popef::Metadata &metadata) {
  for (Tensor *tensor : executablex.getDataStreamTensors()) {
    static constexpr bool isPerReplica = true;
    putAnchorToMetadata(tensor->id,
                        IrLowering::h2dId(tensor->id),
                        isPerReplica,
                        popef::TensorType::INPUT,
                        createTensorInfo(tensor->info),
                        {PopPrograms::ProgramIndex::Program},
                        metadata);
  }
}

/**
 * Serializes anchor(outputs) tensors as popef anchors(metadata about in or out
 * tensor).
 *
 * \param executablex The final executable which contains all the data, metadata
 *                    and configuration parameters necessary to start running
 *                    the program on the device.
 * \param metadata Data needed to run a model using Model Runtime.
 */
void serializeAnchorsAsPopefAnchors(
    const popart::popx::Executablex &executablex,
    popef::Metadata &metadata) {
  for (Tensor *tensor : executablex.getAnchorTensors()) {
    static constexpr bool isPerReplica   = true;
    static constexpr bool isAnchorStream = true;
    putAnchorToMetadata(tensor->id,
                        IrLowering::d2hId(tensor->id, isAnchorStream),
                        isPerReplica,
                        popef::TensorType::OUTPUT,
                        createTensorInfo(tensor->info),
                        {PopPrograms::ProgramIndex::Program},
                        metadata);
  }
}

/**
 * Serializes rng state buffer as popef anchors(metadata about in or out
 * tensor) and its data as TensorBlobs.
 *
 * \param executablex The final executable which contains all the data, metadata
 *                    and configuration parameters necessary to start running
 *                    the program on the device.
 * \param metadata Data needed to run a model using Model Runtime.
 * \param writer Write popef blobs to a given stream.
 */
void serializeRngStateAsPopefAnchors(
    const popart::popx::Executablex &executablex,
    const RngStateBufferType &rngBuffer,
    popef::Metadata &metadata,
    popef::Writer &writer) {
  static constexpr bool isPerReplica      = true;
  static constexpr const char *tensorName = "rngStateTensor";
  const int repFactor                     = metadata.replicationFactor();

  const auto &ir         = executablex.lowering().ir();
  const auto &deviceInfo = *executablex.lowering().getDeviceInfo();

  // popart::DeviceInfo object is used to calculate rng state tensor shape
  // (instead of snap::Graph) because snap::Graph might not exist when
  // we are using deserialized executable. Note that poplar::Target in
  // DeviceInfo contains info about all replicas and poplar::Target in
  // snap::Graph about one replica.
  const std::vector<size_t> tensorShape =
      RngStateLowering::getCombinedRngStateTensorShape(deviceInfo, repFactor);
  const popef::DataType dt = popef::DataType::U32;
  const popart::Shape shape(tensorShape.begin(), tensorShape.end());
  const popef::TensorInfo tensorInfo = createTensorInfo(dt, shape);

  // Tensor has to be handled as an input and output stream
  // as there are programs responsible for loading and storing
  // rng state.
  putAnchorToMetadata(tensorName,
                      IrLowering::h2dId(tensorName),
                      isPerReplica,
                      popef::TensorType::INPUT,
                      tensorInfo,
                      {PopPrograms::ProgramIndex::RngStateFromHost},
                      metadata);
  putAnchorToMetadata(tensorName,
                      std::string("d2h_") + tensorName,
                      isPerReplica,
                      popef::TensorType::OUTPUT,
                      tensorInfo,
                      {PopPrograms::ProgramIndex::RngStateToHost},
                      metadata);

  const bool serializationResult = serializeRngStateAsPopefTensor(
      tensorName, repFactor, dt, shape, rngBuffer, writer);
  if (!serializationResult) {
    std::string warningMessage = "Rng state buffer was not serialized.";
    if (deviceInfo.getType() == DeviceType::OfflineIpu) {
      warningMessage += "You used \"enableLoadAndOffloadRNGState\" option and "
                        "\"OfflineIPU\" mode during compilation.";
    } else if (!ir.getSessionOptions().compileEngine)
      warningMessage += "You did not compile poplar Engine.";
    else {
      warningMessage += "You did not load poplar Engine.";
    }
    warningMessage =
        "Remember that if you would like to run the model using the model "
        "runtime then you have to create your own buffer and callback in your "
        "model runtime application for {}.";
    logging::session::warn(warningMessage, tensorName);
  }
}

/**
 * Serializes random related tensors and buffers(random seed, rng state) as
 * popef anchors(metadata about in or out tensors) and its data as TensorBlobs.
 *
 * \param executablex The final executable which contains all the data, metadata
 *                    and configuration parameters necessary to start running
 *                    the program on the device.
 * \param rngBuffer Content of rng state buffer for all replicas.
 * \param metadata Data needed to run a model using Model Runtime.
 * \param writer Write popef blobs to a given stream.
 */
void serializeRandomTensorsAsPopefAnchors(
    const popart::popx::Executablex &executablex,
    const RngStateBufferType &rngBuffer,
    popef::Metadata &metadata,
    popef::Writer &writer) {
  const auto &ir = executablex.lowering().ir();

  if (ir.getRequiresRandomSeed() && !ir.useSyntheticData()) {
    if (executablex.getSeedTensor() == nullptr) {
      throw error("Seed tensor does not exist.");
    }

    // Tensor has to be handled as an input and output stream
    // as there are programs responsible for loading and storing
    // random seed.
    static constexpr bool isPerReplica = false;
    const popart::Tensor &tensor       = *executablex.getSeedTensor();
    const popef::TensorInfo tensorInfo = createTensorInfo(tensor.info);
    const popef::Anchor anchor =
        putAnchorToMetadata(tensor.id,
                            IrLowering::h2dId(tensor.id),
                            isPerReplica,
                            popef::TensorType::INPUT,
                            tensorInfo,
                            {PopPrograms::ProgramIndex::RandomSeedFromHost},
                            metadata);
    metadata.setSeedHandle(anchor.handle());
    putAnchorToMetadata(tensor.id,
                        "d2h_randomSeed",
                        isPerReplica,
                        popef::TensorType::OUTPUT,
                        tensorInfo,
                        {PopPrograms::ProgramIndex::RandomSeedToHost},
                        metadata);

    serializePopefTensor(tensor, anchor.tensorInfo(), writer);

    if (ir.getSessionOptions().enableLoadAndOffloadRNGState) {
      serializeRngStateAsPopefAnchors(executablex, rngBuffer, metadata, writer);
    }
  }
}

/**
 * Serializes cycle counters(tensors that contains e.g. the number of device
 * cycles (of a single tile, on a single IPU) that your main program takes to
 * execute.) as popef anchors(metadata about in or out tensors).
 *
 * \param executablex The final executable which contains all the data, metadata
 *                    and configuration parameters necessary to start running
 *                    the program on the device.
 * \param metadata Data needed to run a model using Model Runtime.
 */
void serializeCycleCountersAsPopefAnchors(
    const popart::popx::Executablex &executablex,
    popef::Metadata &metadata) {
  const auto &ir = executablex.lowering().ir();

  if (ir.getSessionOptions().instrumentWithHardwareCycleCounter) {
    static constexpr bool isPerReplica = true;
    const auto cycleCountIds = executablex.lowering().getCycleCountIds();
    const popef::DataType dt = popef::DataType::U64;
    const popart::Shape shape(1, 0);
    for (const auto &tid : cycleCountIds) {
      putAnchorToMetadata(tid,
                          IrLowering::cycleCountStreamId(tid),
                          isPerReplica,
                          popef::TensorType::OUTPUT,
                          createTensorInfo(dt, shape),
                          {PopPrograms::ProgramIndex::CycleCountTensorToHost},
                          metadata);
    }
  }
}

/**
 * Serializes needed tensors to the popef file from popart program data.
 * It serializes metadata related to these tensors(anchors in popef metadata)
 * and tensor contents(if needed) as tensor blobs.
 *
 * \param executablex The final executable which contains all the data, metadata
 *                    and configuration parameters necessary to start running
 *                    the program on the device.
 * \param rngBuffer Content of rng state buffer for all replicas.
 * \param metadata Data needed to run a model using Model Runtime.
 * \param writer Write popef blobs to a given stream.
 */
void serializePopefAnchors(const popart::popx::Executablex &executablex,
                           const RngStateBufferType &rngBuffer,
                           popef::Metadata &metadata,
                           popef::Writer &writer) {
  serializeWeightsAsPopefAnchors(executablex, metadata, writer);
  serializeOptimizersAsPopefAnchors(executablex, metadata, writer);
  serializeDataStreamsAsPopefAnchors(executablex, metadata);
  serializeAnchorsAsPopefAnchors(executablex, metadata);
  serializeRandomTensorsAsPopefAnchors(
      executablex, rngBuffer, metadata, writer);
  serializeCycleCountersAsPopefAnchors(executablex, metadata);
}

static const std::unordered_map<int64_t, std::string> &
programsMap(const popart::popx::Executablex &executablex) {
  const auto &customPrograms =
      executablex.lowering().getProgramHandleIndexMap();
  if (customPrograms.empty())
    return PopPrograms::commonPrograms;

  static std::unordered_map<int64_t, std::string> out(
      PopPrograms::commonPrograms);

  std::transform(customPrograms.cbegin(),
                 customPrograms.cend(),
                 std::inserter(out, out.end()),
                 [](const std::pair<std::string, unsigned> &p) {
                   return std::make_pair(p.second, p.first);
                 });

  return out;
}

/**
 * Serializes needed data to run poplar executable created by popart
 * using Model Runtime. Data mainly comes from executablex object with
 * one minor exception: rngBuffer which comes from devicex object.
 * This includes: popef::Metadata, tensor blobs(content of the tensors).
 *
 * \param executablex The final executable which contains all the data, metadata
 *                    and configuration parameters necessary to start running
 *                    the program on the device.
 * \param rngBuffer Content of rng state buffer for all replicas.
 * \param metadata Data needed to run a model using Model Runtime.
 * \param writer Write popef blobs to a given stream.
 */
void serializePopefMetadata(const popart::popx::Executablex &executablex,
                            const RngStateBufferType &rngBuffer,
                            const std::string &programHash,
                            popef::Writer &writer) {
  const auto &ir_lowering = executablex.lowering();
  const auto &ir          = ir_lowering.ir();
  const auto &opts        = ir.getSessionOptions();

  const bool isInference =
      ir.getExecutionMode() == Ir::ExecutionMode::Inference;
  const int64_t numProcs =
      opts.enableDistributedReplicatedGraphs ? opts.globalReplicationFactor : 1;

  popef::Metadata metadata;
  metadata.setReplicationFactor(ir_lowering.getReplicationFactor());
  metadata.setNumIpus(ir_lowering.getDeviceInfo()->getNumIpus());
  metadata.setIpuVersion(getIpuVersion(*ir_lowering.getDeviceInfo()));
  metadata.setExecutable(programHash);
  metadata.setIsPOD(isPOD(*ir_lowering.getDeviceInfo()));
  metadata.setNumProcesses(numProcs);
  metadata.setIsInference(isInference);
  metadata.setEngineOptions(
      convertOptionFlagsToOptions(ir_lowering.engineOptions));
  metadata.setDeviceOptions(convertOptionFlagsToOptions(
      ir_lowering.getDeviceInfo()->getOptionFlags()));

  serializePopefAnchors(executablex, rngBuffer, metadata, writer);

  std::vector<int64_t> loadPrograms{
      PopPrograms::ProgramIndex::WeightsFromHost,
      PopPrograms::ProgramIndex::OptimizerFromHost};
  std::vector<int64_t> mainPrograms{PopPrograms::ProgramIndex::Program};
  std::vector<int64_t> savePrograms{PopPrograms::ProgramIndex::WeightsToHost};

  if (!metadata.seedHandle().empty()) {
    loadPrograms.push_back(PopPrograms::ProgramIndex::RandomSeedFromHost);
    savePrograms.push_back(PopPrograms::ProgramIndex::RandomSeedToHost);

    if (ir.getSessionOptions().enableLoadAndOffloadRNGState) {
      loadPrograms.push_back(PopPrograms::ProgramIndex::RngStateFromHost);
      savePrograms.push_back(PopPrograms::ProgramIndex::RngStateToHost);
    }
  }

  if (ir.getSessionOptions().instrumentWithHardwareCycleCounter) {
    savePrograms.push_back(PopPrograms::ProgramIndex::CycleCountTensorToHost);
  }

  popef::ProgramFlow flow;
  flow.setLoad(loadPrograms);
  flow.setMain(mainPrograms);
  flow.setSave(savePrograms);
  metadata.setProgramFlow(flow);

  metadata.setProgramsMap(programsMap(executablex));

  writer.write(metadata);
}
} // namespace

/** To see description go to the function declaration. */
void serializeEngineExecutable(std::ostream &out,
                               const popart::popx::Devicex &device) {
  const std::string programHash = std::to_string(device.ir().getHash());
  popef::Writer popefWriter(out);

  // Export Popart specific data.
  std::shared_ptr<popef::BlobWriter> popefOpaque =
      popefWriter.createOpaqueBlob(popartOpaqueName, programHash);
  serializePopartExecutable(popefOpaque->stream, device.executable_);

  try {
    // Export popef metadata and tensor contents.
    serializePopefMetadata(
        device.executable_, device.rngBuffer, programHash, popefWriter);
  } catch (const std::exception &e) {
    logging::session::warn(
        "Serializing popef metadata ended with failure due to {}. The popef "
        "file cannot be used to run model using model_runtime",
        e.what());
  }

  // Export Poplar engine's executable.
  if (device.pEngine.get()) {
    static constexpr bool compress = false;
    std::shared_ptr<popef::BlobWriter> popefExe =
        popefWriter.createExecutable(programHash, compress);
    device.pEngine.get()->serializeExecutable(popefExe->stream);
  }
}

/** Implementation of Reader class. */
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
      : popefReader(setupReader(in)), popartOpaque(findPopartOpaque()),
        poplarExecutable(findPoplarExecutable()),
        popefMetadata(findPopefMetadata()), tensorData(findPopefTensors()),
        hash(getExecutableHash()) {}

  popef::Reader popefReader;
  const OpaqueReaderOpt popartOpaque;
  const ExecReaderOpt poplarExecutable;
  const MetadataOpt popefMetadata;
  const TensorReaderVec tensorData;
  const size_t hash;

private:
  /**
   * Creates \c popef::Reader object and parses stream that contains
   * popef file.
   *
   * \param in Stream to be parsed that contains popef file.
   * \return Object of \c popef::Reader class with parsed popef stream.
   */
  popef::Reader setupReader(std::shared_ptr<std::istream> in) {
    popef::Reader reader;
    reader.parseStream(in);
    return reader;
  }

  /**
   * \return Opaque blob which contains serialized popart specific data.
   */
  OpaqueReaderOpt findPopartOpaque() {
    auto popartOpaqueMatcher = [](const popef::OpaqueReader &opaque) {
      return opaque.name.find(popartOpaqueName) != std::string::npos;
    };

    const std::vector<popef::OpaqueReader> &opaques = popefReader.opaqueBlobs();
    const int numOfMatchedPopartOpaques =
        std::count_if(opaques.begin(), opaques.end(), popartOpaqueMatcher);
    if (numOfMatchedPopartOpaques > 1) {
      throw error("The file contains more than one Popart related opaque in "
                  "the popef file.");
    }
    OpaqueReaderIt opaqueIt =
        std::find_if(opaques.begin(), opaques.end(), popartOpaqueMatcher);

    const bool opaqueExists = opaqueIt != opaques.end();
    const OpaqueReaderOpt opaqueReader =
        opaqueExists ? OpaqueReaderOpt(*opaqueIt) : nonstd::nullopt;
    return opaqueReader;
  }

  /**
   * \return Poplar executable blob.
   */
  ExecReaderOpt findPoplarExecutable() {
    const std::vector<popef::ExecutableReader> &execs =
        popefReader.executables();

    ExecReaderIt execIt = execs.end();
    if (popartOpaque.has_value()) {
      auto poplarExecMatcher =
          [this](const popef::ExecutableReader &executable) {
            return executable.name == popartOpaque->get().executable;
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

  /**
   * \return Popef metadata.
   */
  MetadataOpt findPopefMetadata() {
    auto metadataMatcher = [this](const popef::Metadata &metadata) {
      if (!popartOpaque.has_value())
        return false;
      return metadata.executable() == popartOpaque->get().executable;
    };

    const std::vector<popef::Metadata> &metadataVec = popefReader.metadata();
    const int numOfMatchedPopefMetadata =
        std::count_if(metadataVec.begin(), metadataVec.end(), metadataMatcher);
    if (numOfMatchedPopefMetadata > 1) {
      throw error("The file contains more than one Popef metadata");
    }
    MetadataIt metadataIt =
        std::find_if(metadataVec.begin(), metadataVec.end(), metadataMatcher);

    const bool metadataExists = metadataIt != metadataVec.end();
    return metadataExists ? MetadataOpt(*metadataIt) : nonstd::nullopt;
  }

  /**
   * \return All tensor blobs associated with anchors from popef metadata.
   */
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

  /**
   * \return The executable hash or 0 if the stream contains
   *         corrupted data.
   */
  size_t getExecutableHash() const {
    size_t hash = 0;

    if (poplarExecutable.has_value() || popartOpaque.has_value()) {
      const std::string &hashString = poplarExecutable.has_value()
                                          ? poplarExecutable->get().name
                                          : popartOpaque->get().executable;
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

/** To see description go to the function declaration. */
Reader::Reader(std::shared_ptr<std::istream> in)
    : _impl(std::make_unique<ReaderImpl>(in)) {}
Reader::Reader(Reader &&reader) : _impl(std::move(reader._impl)) {}
Reader::~Reader() = default;

/** To see description go to the function declaration. */
size_t Reader::readExecutableHash() const { return _impl->hash; }

/** To see description go to the function declaration. */
bool Reader::containsPoplarExecutable() const {
  return _impl->poplarExecutable.has_value();
}

/** To see description go to the function declaration. */
bool Reader::containsExecutable() const {
  return _impl->popartOpaque.has_value();
}

/** To see description go to the function declaration. */
bool Reader::containsPopefMetadata() {
  return _impl->popefMetadata.has_value();
}

/** To see description go to the function declaration. */
poplar::Executable Reader::deserializePoplarExecutable() const {
  if (!containsPoplarExecutable()) {
    throw error("The file does not contain poplar executable.");
  }

  const popef::ExecutableReader &exeReader = _impl->poplarExecutable->get();
  return poplar::Executable::deserialize(
      exeReader.getStandaloneExecutableStream());
}

/** To see description go to the function declaration. */
std::unique_ptr<popart::popx::Executablex>
Reader::deserializeExecutable(popart::Ir &ir,
                              popart::popx::IrLowering &lowering) const {
  if (!containsExecutable()) {
    throw error("The file does not contain popart metadata.");
  }

  const popef::OpaqueReader &opaqueReader = _impl->popartOpaque->get();
  std::unique_ptr<std::istream> opaqueStream(
      opaqueReader.getStandaloneDataStream());
  auto executablex = deserializePopartExecutable(*opaqueStream, ir, lowering);

  return executablex;
}

} // namespace serialization
} // namespace popx
} // namespace popart
