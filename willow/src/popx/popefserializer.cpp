// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/popx/popefserializer.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <popef/Reader.hpp>
#include <popef/Types.hpp>
#include <popef/Writer.hpp>

#include <snap/Graph.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Executable.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/StringRef.hpp>
#include <poplar/Target.hpp>
#include <popx/executablexserializer.hpp>
#include <popx/rng/rngstatelowering.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/popprograms.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {
namespace popx {
namespace serialization {
namespace {

const std::string popartOpaqueName("popart");

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
  auto popartMetadata =
      deserializePopartExecutable(*opaque_stream, ir, lowering);

  return popartMetadata;
}

} // namespace serialization
} // namespace popx
} // namespace popart
