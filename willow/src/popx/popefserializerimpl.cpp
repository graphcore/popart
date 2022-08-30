// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "popefserializerimpl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <popef/Reader.hpp>
#include <popef/Types.hpp>
#include <popef/Writer.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Executable.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/StringRef.hpp>
#include <poplar/Target.hpp>

#include "popart/datatype.hpp"
#include "popart/devicemanager.hpp"
#include "popart/error.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/devicex.hpp"
#include "popart/popx/executablex.hpp"
#include "popart/popx/irlowering.hpp"
#include "popart/popx/popprograms.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordata.hpp"
#include "popart/tensorinfo.hpp"
#include "popx/executablexserializer.hpp"
#include "popx/rng/rngstatelowering.hpp"

namespace popart {
namespace popx {
namespace serialization {

static constexpr const char *popartOpaqueName = "popart";

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
    errorStream << "There is no PopEF mapping for popart::DataType " << type;
    throw error(errorStream.str());
  }
}

popef::TensorInfo createTensorInfo(const popef::DataType dt,
                                   const std::vector<int64_t> &shape) {
  popef::TensorInfo tensorInfo;
  tensorInfo.setDataType(dt);
  tensorInfo.setShape(shape);
  return tensorInfo;
}

popef::TensorInfo createTensorInfo(const popart::TensorInfo &info) {
  popef::TensorInfo tensorInfo;
  const popef::DataType dt = toPopefDataType(info.getDataTypeInfo()->type());
  return createTensorInfo(dt, info.shape());
}

WriterImpl::WriterImpl(std::ostream &out, const popart::popx::Devicex &device)
    : _writer(out), _executablex(device.executable_),
      popartPrograms(createProgramsMap()), _engine(device.pEngine),
      _programHash(std::to_string(_executablex.ir().getHash())),
      _rngBuffer(device.rngBuffer) {
  createAnchors();
}

/*static*/ void
WriterImpl::serializePopefTensor(const popart::Tensor &tensor,
                                 const popef::TensorInfo &tensorInfo,
                                 popef::Writer &writer) {
  if (!tensor.hasTensorData()) {
    throw error("Tensor {} has no allocated data.", tensor.id);
  }

  if (tensor.tensorData()->size() != tensorInfo.sizeInBytes()) {
    throw error("Cannot save tensor {} to the file. Its size differs from the "
                "expectations. Expected size {}, tensor size {}.",
                tensor.id,
                tensorInfo.sizeInBytes(),
                tensor.tensorData()->size());
  }

  popef::TensorDataInfo tensorDataInfo;
  tensorDataInfo.setName(tensor.id);
  tensorDataInfo.setTensorInfo(tensorInfo);

  std::shared_ptr<popef::BlobWriter> tensorWriter =
      writer.createTensorData(tensorDataInfo);
  const char *dataPtr =
      reinterpret_cast<const char *>(tensor.tensorData()->data());
  tensorWriter->stream.write(dataPtr, tensorInfo.sizeInBytes());
}

void WriterImpl::serializeTensorData() {
  for (const auto &anchor : _anchorsWithData) {
    if (anchor.name() == rngStateTensorName) {
      serializeRngBufferContent(anchor);
    } else {
      const popart::Tensor *tensor = _executablex.getTensor(anchor.name());
      serializePopefTensor(*tensor, anchor.tensorInfo(), _writer);
    }
  }
}

void WriterImpl::serializePopartMetadata() {
  std::shared_ptr<popef::BlobWriter> popefOpaque =
      _writer.createOpaqueBlob(popartOpaqueName, _programHash);
  serializePopartExecutable(popefOpaque->stream, _executablex);
}

void WriterImpl::serializePoplarEngine() {
  if (_engine.get()) {
    static constexpr bool compress = false;
    std::shared_ptr<popef::BlobWriter> popefExe =
        _writer.createExecutable(_programHash, compress);
    _engine.get()->serializeExecutable(popefExe->stream);
  }
}

void WriterImpl::serializePopefMetadata() {
  const auto &ir_lowering = _executablex.lowering();
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
  metadata.setExecutable(_programHash);
  metadata.setIsPOD(isPOD(*ir_lowering.getDeviceInfo()));
  metadata.setNumProcesses(numProcs);
  metadata.setIsInference(isInference);
  metadata.setEngineOptions(
      convertOptionFlagsToOptions(ir_lowering.engineOptions));
  metadata.setDeviceOptions(convertOptionFlagsToOptions(
      ir_lowering.getDeviceInfo()->getOptionFlags()));
  metadata.setSeedHandle(_seedHandle);

  std::vector<popef::Anchor> &metadataAnchors = metadata.anchors();
  metadataAnchors.insert(metadataAnchors.begin(),
                         _anchorsWithoutData.begin(),
                         _anchorsWithoutData.end());
  metadataAnchors.insert(
      metadataAnchors.end(), _anchorsWithData.begin(), _anchorsWithData.end());

  using ProgramIndexType = popef::ProgramFlow::ProgramIndexType;
  using ProgramIndex     = PopPrograms::ProgramIndex;
  std::vector<ProgramIndexType> loadPrograms{ProgramIndex::WeightsFromHost};
  std::vector<ProgramIndexType> mainPrograms{ProgramIndex::Program};
  std::vector<ProgramIndexType> savePrograms{};

  if (!isInference) {
    loadPrograms.push_back(ProgramIndex::OptimizerFromHost);
    savePrograms.push_back(ProgramIndex::WeightsToHost);
  }

  if (!metadata.seedHandle().empty()) {
    loadPrograms.push_back(ProgramIndex::RandomSeedFromHost);
    savePrograms.push_back(ProgramIndex::RandomSeedToHost);

    if (ir.getSessionOptions().enableLoadAndOffloadRNGState) {
      loadPrograms.push_back(ProgramIndex::RngStateFromHost);
      savePrograms.push_back(ProgramIndex::RngStateToHost);
    }
  }

  if (ir.getSessionOptions().instrumentWithHardwareCycleCounter) {
    savePrograms.push_back(ProgramIndex::CycleCountTensorToHost);
  }

  popef::ProgramFlow flow;
  flow.setLoad(loadPrograms);
  flow.setMain(mainPrograms);
  flow.setSave(savePrograms);
  metadata.setProgramFlow(flow);

  metadata.setProgramsMap(popartPrograms);

  _writer.write(metadata);
}

const std::unordered_map<popef::ProgramFlow::ProgramIndexType, std::string>
WriterImpl::createProgramsMap() const {
  const auto &customPrograms =
      _executablex.lowering().getProgramHandleIndexMap();
  if (customPrograms.empty())
    return PopPrograms::commonPrograms;

  std::unordered_map<popef::ProgramFlow::ProgramIndexType, std::string> out(
      PopPrograms::commonPrograms);

  std::transform(customPrograms.cbegin(),
                 customPrograms.cend(),
                 std::inserter(out, out.end()),
                 [](const std::pair<std::string, unsigned> &p) {
                   return std::make_pair(p.second, p.first);
                 });

  return out;
}

/*static*/ popef::Anchor WriterImpl::createAnchor(
    const std::string &name,
    const std::string &handle,
    const popef::TensorInfo &tensorInfo,
    const bool isPerReplica,
    const popef::TensorType type,
    const std::vector<popef::ProgramFlow::ProgramIndexType> &programs) {
  popef::Anchor anchor;

  anchor.setName(name);
  anchor.setHandle(handle);
  anchor.setTensorInfo(tensorInfo);
  anchor.setIsPerReplica(isPerReplica);
  anchor.setType(type);
  anchor.setPrograms(programs);

  return anchor;
}

void WriterImpl::addDataInputAnchor(
    const std::string &name,
    const std::string &handle,
    const popef::TensorInfo &tensorInfo,
    const bool isPerReplica,
    const std::vector<popef::ProgramFlow::ProgramIndexType> &programs) {
  _anchorsWithData.push_back(createAnchor(name,
                                          handle,
                                          tensorInfo,
                                          isPerReplica,
                                          popef::TensorType::INPUT,
                                          programs));
}

void WriterImpl::addInputAnchor(
    const std::string &name,
    const std::string &handle,
    const popef::TensorInfo &tensorInfo,
    const bool isPerReplica,
    const std::vector<popef::ProgramFlow::ProgramIndexType> &programs) {
  _anchorsWithoutData.push_back(createAnchor(name,
                                             handle,
                                             tensorInfo,
                                             isPerReplica,
                                             popef::TensorType::INPUT,
                                             programs));
}

void WriterImpl::addOutputAnchor(
    const std::string &name,
    const std::string &handle,
    const popef::TensorInfo &tensorInfo,
    const bool isPerReplica,
    const std::vector<popef::ProgramFlow::ProgramIndexType> &programs) {
  _anchorsWithoutData.push_back(createAnchor(name,
                                             handle,
                                             tensorInfo,
                                             isPerReplica,
                                             popef::TensorType::OUTPUT,
                                             programs));
}

void WriterImpl::addDataUnknownAnchor(const std::string &name,
                                      const popef::TensorInfo &tensorInfo,
                                      bool isPerReplica) {
  _anchorsWithData.push_back(createAnchor(
      name, "", tensorInfo, isPerReplica, popef::TensorType::UNKNOWN, {}));
}

void WriterImpl::createPopefAnchorsFromWeights() {
  static constexpr bool isAnchorStream = false;
  const auto &irLowering               = _executablex.lowering();
  const auto &ir                       = irLowering.ir();

  for (auto *tensor : _executablex.getWeightTensors()) {
    if (tensor->hasProducer()) {
      throw error("Weights are tensors of variable type that do not"
                  "have producers.");
    }

    const popef::DataType dt      = toPopefDataType(tensor->info.dataType());
    const popart::Shape hostShape = tensor->getVariableSettings().shapeOnHost(
        tensor->info.shape(), irLowering.getReplicationFactor());
    const popef::TensorInfo tensorInfo = createTensorInfo(dt, hostShape);

    const bool isPerReplica = hostShape != tensor->info.shape();

    if (!ir.streamingIsDisabledForTensor(tensor)) {
      // Tensor has to be handled as an input and output stream
      // as there are programs responsible for loading and storing
      // weights.
      addDataInputAnchor(tensor->id,
                         IrLowering::h2dId(tensor->id),
                         tensorInfo,
                         isPerReplica,
                         {PopPrograms::ProgramIndex::WeightsFromHost});

      addOutputAnchor(tensor->id,
                      IrLowering::d2hId(tensor->id, isAnchorStream),
                      tensorInfo,
                      isPerReplica,
                      {PopPrograms::ProgramIndex::WeightsToHost});
    } else {
      addDataUnknownAnchor(tensor->id, tensorInfo, isPerReplica);
    }
  }
}

void WriterImpl::createPopefAnchorsFromOptimizers() {
  for (auto *tensor : _executablex.getOptimizerTensors()) {
    static constexpr bool isPerReplica = false;
    popef::Anchor anchor;
    anchor.setName(tensor->id);
    anchor.setHandle(IrLowering::h2dId(tensor->id));
    anchor.setTensorInfo(createTensorInfo(tensor->info));
    anchor.setIsPerReplica(isPerReplica);
    anchor.setType(popef::TensorType::INPUT);
    anchor.setPrograms({PopPrograms::ProgramIndex::OptimizerFromHost});
    _anchorsWithData.push_back(anchor);
  }
}

void WriterImpl::createPopefAnchorsFromDataStreams() {
  const auto &irLowering = _executablex.lowering();

  for (Tensor *tensor : _executablex.getDataStreamTensors()) {
    const bool isPerReplica          = irLowering.getReplicationFactor() > 1;
    const popart::Shape &tensorShape = tensor->info.shape();

    popart::Shape shape(tensorShape.begin(), tensorShape.end());
    if (isPerReplica) {
      shape.insert(shape.begin(), irLowering.getReplicationFactor());
    }
    const popef::DataType dt = toPopefDataType(tensor->info.dataType());
    const popef::TensorInfo tensorInfo = createTensorInfo(dt, shape);

    addInputAnchor(tensor->id,
                   IrLowering::h2dId(tensor->id),
                   tensorInfo,
                   isPerReplica,
                   {PopPrograms::ProgramIndex::Program});
  }
}

void WriterImpl::createPopefAnchorsFromAnchors() {
  static constexpr bool isAnchorStream = true;
  const auto &irLowering               = _executablex.lowering();

  for (Tensor *tensor : _executablex.getAnchorTensors()) {
    const bool isPerReplica          = irLowering.getReplicationFactor() > 1;
    const popart::Shape &tensorShape = tensor->info.shape();

    popart::Shape shape(tensorShape.begin(), tensorShape.end());
    if (isPerReplica) {
      shape.insert(shape.begin(), irLowering.getReplicationFactor());
    }
    const popef::DataType dt = toPopefDataType(tensor->info.dataType());
    const popef::TensorInfo tensorInfo = createTensorInfo(dt, shape);

    addOutputAnchor(tensor->id,
                    IrLowering::d2hId(tensor->id, isAnchorStream),
                    tensorInfo,
                    isPerReplica,
                    {PopPrograms::ProgramIndex::Program});
  }
}

void WriterImpl::createPopefAnchorsFromRandomSeed() {
  if (_executablex.getSeedTensor() == nullptr) {
    throw error("Seed tensor does not exist.");
  }

  // Tensor has to be handled as an input and output stream
  // as there are programs responsible for loading and storing
  // random seed.
  static constexpr bool isPerReplica = false;
  const popart::Tensor &tensor       = *_executablex.getSeedTensor();
  const popef::TensorInfo tensorInfo = createTensorInfo(tensor.info);
  const std::string inputSeedHandle  = IrLowering::h2dId(tensor.id);
  addDataInputAnchor(tensor.id,
                     inputSeedHandle,
                     tensorInfo,
                     isPerReplica,
                     {PopPrograms::ProgramIndex::RandomSeedFromHost});
  _seedHandle = inputSeedHandle;

  addOutputAnchor(tensor.id,
                  "d2h_randomSeed",
                  tensorInfo,
                  isPerReplica,
                  {PopPrograms::ProgramIndex::RandomSeedToHost});
}

void WriterImpl::createPopefAnchorsFromRNGState() {
  const auto &irLowering   = _executablex.lowering();
  const unsigned repFactor = irLowering.getReplicationFactor();
  const bool isPerReplica  = repFactor > 1;
  const auto &deviceInfo   = *irLowering.getDeviceInfo();

  // popart::DeviceInfo object is used to calculate rng state tensor shape
  // (instead of snap::Graph) because snap::Graph might not exist when
  // we are using deserialized executable. Note that poplar::Target in
  // DeviceInfo contains info about all replicas and poplar::Target in
  // snap::Graph about one replica.
  const std::vector<size_t> tensorShape =
      RngStateLowering::getCombinedRngStateTensorShape(deviceInfo, repFactor);
  popart::Shape shape(tensorShape.begin(), tensorShape.end());
  if (isPerReplica) {
    shape.insert(shape.begin(), repFactor);
  }
  const popef::DataType dt           = popef::DataType::U32;
  const popef::TensorInfo tensorInfo = createTensorInfo(dt, shape);

  // Tensor has to be handled as an input and output stream
  // as there are programs responsible for loading and storing
  // rng state.
  addDataInputAnchor(rngStateTensorName,
                     std::string("h2d_") + rngStateTensorName,
                     tensorInfo,
                     isPerReplica,
                     {PopPrograms::ProgramIndex::RngStateFromHost});

  addOutputAnchor(rngStateTensorName,
                  std::string("d2h_") + rngStateTensorName,
                  tensorInfo,
                  isPerReplica,
                  {PopPrograms::ProgramIndex::RngStateToHost});
}

void WriterImpl::createPopefAnchorsFromCycleCounters() {
  const auto &irLowering   = _executablex.lowering();
  const bool isPerReplica  = irLowering.getReplicationFactor() > 1;
  const auto cycleCountIds = irLowering.getCycleCountIds();
  const popef::DataType dt = popef::DataType::U64;
  const popart::Shape shape =
      isPerReplica ? popart::Shape(1, irLowering.getReplicationFactor())
                   : popart::Shape();
  for (const auto &tid : cycleCountIds) {
    addOutputAnchor(tid,
                    IrLowering::cycleCountStreamId(tid),
                    createTensorInfo(dt, shape),
                    isPerReplica,
                    {PopPrograms::ProgramIndex::CycleCountTensorToHost});
  }
}

void WriterImpl::createAnchors() {
  const auto &ir = _executablex.ir();

  createPopefAnchorsFromWeights();
  createPopefAnchorsFromOptimizers();
  createPopefAnchorsFromDataStreams();
  createPopefAnchorsFromAnchors();

  if (ir.getRequiresRandomSeed() && !ir.useSyntheticData()) {
    createPopefAnchorsFromRandomSeed();
    if (ir.getSessionOptions().enableLoadAndOffloadRNGState) {
      createPopefAnchorsFromRNGState();
    }
  }

  if (ir.getSessionOptions().instrumentWithHardwareCycleCounter) {
    createPopefAnchorsFromCycleCounters();
  }
}

void WriterImpl::serializeRngBufferContent(
    const popef::Anchor &rngBufferAnchor) {
  const auto &irLowering   = _executablex.lowering();
  const auto &ir           = irLowering.ir();
  const auto &deviceInfo   = irLowering.getDeviceInfo();
  const unsigned repFactor = irLowering.getReplicationFactor();

  if (!_rngBuffer.empty()) {
    if (_rngBuffer.size() != repFactor) {
      throw error("RNG State buffer does not have data for all replicas. "
                  "Devicex did not allocate data for all replicas.");
    }

    popef::TensorDataInfo tensorDataInfo;
    tensorDataInfo.setName(rngBufferAnchor.name());
    tensorDataInfo.setTensorInfo(rngBufferAnchor.tensorInfo());
    const int64_t nbytes = tensorDataInfo.tensorInfo().sizeInBytes();
    std::shared_ptr<popef::BlobWriter> tensorWriter =
        _writer.createTensorData(tensorDataInfo);

    for (int i = 0; i < repFactor; i++) {
      const auto rngStateIt = _rngBuffer.find(i);
      if (rngStateIt == _rngBuffer.cend()) {
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
  } else {
    std::string warningMessage = "Rng state buffer was not serialized.";
    if (deviceInfo->getType() == DeviceType::OfflineIpu) {
      warningMessage += "You used \"enableLoadAndOffloadRNGState\" option and "
                        "\"OfflineIPU\" mode during compilation.";
    } else if (!ir.getSessionOptions().compileEngine)
      warningMessage += "You did not compile poplar Engine.";
    else {
      warningMessage += "You did not load poplar Engine.";
    }
    warningMessage +=
        "Remember that if you would like to run the model using the model "
        "runtime then you have to create your own buffer and callback in "
        "your model runtime application for {}.";
    logging::session::warn(warningMessage, rngBufferAnchor.name());
  }
}

/*static*/ int WriterImpl::getIpuVersion(const DeviceInfo &deviceInfo) {
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

/*static*/ bool WriterImpl::isPOD(const DeviceInfo &deviceInfo) {
  const std::string targetSystemStr = "POD";
  const std::string systemVersionStr =
      deviceInfo.getTarget().getTargetSystemString();
  return systemVersionStr.find(targetSystemStr) != std::string::npos;
}

/*static*/ std::vector<popef::Option>
WriterImpl::convertOptionFlagsToOptions(const poplar::OptionFlags &optFlags) {
  std::vector<popef::Option> opts;
  std::transform(optFlags.begin(),
                 optFlags.end(),
                 std::back_inserter(opts),
                 [](const auto &optFlag) {
                   return popef::Option(optFlag.first, optFlag.second);
                 });
  return opts;
}

ReaderImpl::ReaderImpl(const std::vector<std::shared_ptr<std::istream>> &in_vec)
    : _popefReader(setupReader(in_vec)), _popartOpaque(findPopartOpaque()),
      _poplarExecutable(findPoplarExecutable()),
      _popefMetadata(findPopefMetadata()), _tensorDataVec(findPopefTensors()),
      _hash(getExecutableHash()) {}

/*static*/ popef::Reader ReaderImpl::setupReader(
    const std::vector<std::shared_ptr<std::istream>> &in_vec) {
  popef::Reader reader;
  for (const auto &in : in_vec) {
    reader.parseStream(in);
  }
  return reader;
}

poplar::Executable ReaderImpl::deserializePoplarExecutable() const {
  if (!_poplarExecutable.has_value()) {
    throw error("The file does not contain poplar executable.");
  }

  const popef::ExecutableReader &exeReader = _poplarExecutable->get();
  return poplar::Executable::deserialize(
      exeReader.getStandaloneExecutableStream());
}

std::unique_ptr<popart::popx::Executablex>
ReaderImpl::deserializeExecutable(popart::Ir &ir,
                                  popart::popx::IrLowering &lowering) const {
  if (!_popartOpaque.has_value()) {
    throw error("The file does not contain popart metadata.");
  }

  const popef::OpaqueReader &opaqueReader = _popartOpaque->get();
  std::unique_ptr<std::istream> opaqueStream(
      opaqueReader.getStandaloneDataStream());
  auto executablex =
      deserializePopartExecutable(*opaqueStream, ir, lowering, _tensorDataVec);

  return executablex;
}

ReaderImpl::OpaqueReaderOpt ReaderImpl::findPopartOpaque() {
  auto popartOpaqueMatcher = [](const popef::OpaqueReader &opaque) {
    return opaque.name.find(popartOpaqueName) != std::string::npos;
  };

  const std::vector<popef::OpaqueReader> &opaques = _popefReader.opaqueBlobs();
  const int numOfMatchedPopartOpaques =
      std::count_if(opaques.begin(), opaques.end(), popartOpaqueMatcher);
  if (numOfMatchedPopartOpaques > 1) {
    throw error("The file contains more than one Popart related opaque in "
                "the PopEF file.");
  }
  OpaqueReaderIt opaqueIt =
      std::find_if(opaques.begin(), opaques.end(), popartOpaqueMatcher);

  const bool opaqueExists = opaqueIt != opaques.end();
  const OpaqueReaderOpt opaqueReader =
      opaqueExists ? OpaqueReaderOpt(*opaqueIt) : nonstd::nullopt;
  return opaqueReader;
}

ReaderImpl::ExecReaderOpt ReaderImpl::findPoplarExecutable() {
  const std::vector<popef::ExecutableReader> &execs =
      _popefReader.executables();

  ExecReaderIt execIt = execs.end();
  if (_popartOpaque.has_value()) {
    auto poplarExecMatcher = [this](const popef::ExecutableReader &executable) {
      return executable.name == _popartOpaque->get().executable;
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
                  "executable does not exist and the PopEF file "
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

ReaderImpl::MetadataOpt ReaderImpl::findPopefMetadata() {
  auto metadataMatcher = [this](const popef::Metadata &metadata) {
    if (!_popartOpaque.has_value())
      return false;
    return metadata.executable() == _popartOpaque->get().executable;
  };

  const std::vector<popef::Metadata> &metadataVec = _popefReader.metadata();
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

std::vector<popef::TensorReader> ReaderImpl::findPopefTensors() {
  auto tensorMatcher = [this](const popef::TensorReader &tensor) {
    if (!_popefMetadata.has_value())
      return false;
    auto anchorMatcher = [&tensor](const popef::Anchor &anchor) {
      return anchor.name() == tensor.info.name();
    };
    const std::vector<popef::Anchor> &anchors = _popefMetadata->get().anchors();
    auto anchorIt = std::find_if(anchors.begin(), anchors.end(), anchorMatcher);
    return anchorIt != anchors.end();
  };

  std::vector<popef::TensorReader> tensors;
  const std::vector<popef::TensorReader> &allTensors = _popefReader.tensors();
  std::copy_if(allTensors.begin(),
               allTensors.end(),
               std::back_inserter(tensors),
               tensorMatcher);

  return tensors;
}

size_t ReaderImpl::getExecutableHash() const {
  size_t _hash = 0;

  if (_poplarExecutable.has_value() || _popartOpaque.has_value()) {
    const std::string &hashString = _poplarExecutable.has_value()
                                        ? _poplarExecutable->get().name
                                        : _popartOpaque->get().executable;
    std::stringstream ss(hashString);
    ss >> _hash;
    if (ss.fail()) {
      throw error("Neither the poplar executable nor the popart metadata "
                  "contains a hash number.");
    }
  }

  return _hash;
}

} // namespace serialization
} // namespace popx
} // namespace popart
