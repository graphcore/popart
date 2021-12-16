// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <algorithm>

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

  throw error("Invalid TensorType {}", static_cast<int>(type));
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

  throw error("Invalid TensorType {}", static_cast<int>(type));
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

  throw error("Invalid DataType {}", static_cast<int>(type));
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

  throw error("Invalid DataType {}", static_cast<int>(type));
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

void serializeExecutable(std::ostream &out,
                         const poplar::Executable *poplarExecutable,
                         const popart::popx::Executablex *executable,
                         size_t hash) {
  const std::string programHash = std::to_string(hash);
  popef::Writer popefWriter(out);

  // Export Popart specific data
  if (executable) {
    std::shared_ptr<popef::BlobWriter> popefOpaque =
        popefWriter.createOpaqueBlob(popartOpaqueName, programHash);
    serializePopartExecutable(popefOpaque->stream, *executable);
  }

  // Export Poplar executable
  if (poplarExecutable) {
    static constexpr bool compress = false;
    std::shared_ptr<popef::BlobWriter> popefExe =
        popefWriter.createExecutable(programHash, compress);
    poplarExecutable->serialize(popefExe->stream);
  }
}

class ReaderImpl {
public:
  template <typename T>
  using optional_ref    = nonstd::optional<std::reference_wrapper<T>>;
  using OpaqueReaderOpt = optional_ref<const popef::OpaqueReader>;
  using ExecReaderOpt   = optional_ref<const popef::ExecutableReader>;
  using OpaqueReaderIt  = std::vector<popef::OpaqueReader>::const_iterator;
  using ExecReaderIt    = std::vector<popef::ExecutableReader>::const_iterator;

  ReaderImpl(const std::istream &in)
      : popefReader(setupReader(in)), popartMetadata(findPopartMetadata()),
        poplarExecutable(findPoplarExecutable()), hash(getExecutableHash()) {}

  popef::Reader popefReader;
  OpaqueReaderOpt popartMetadata;
  ExecReaderOpt poplarExecutable;
  size_t hash;

private:
  popef::Reader setupReader(const std::istream &in) {
    auto in_ptr = std::make_shared<std::istream>(in.rdbuf());
    popef::Reader reader;
    reader.parseStream(in_ptr);
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
    return opaqueExists ? OpaqueReaderOpt(*opaqueIt) : nonstd::nullopt;
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

  size_t getExecutableHash() {
    size_t hash = 0;

    if (!poplarExecutable.has_value() && !popartMetadata.has_value()) {
      throw error(
          "The file contains neither poplar executable nor popart metadata.");
    }

    const std::string &hashString = poplarExecutable.has_value()
                                        ? poplarExecutable->get().name
                                        : popartMetadata->get().executable;

    std::stringstream ss(hashString);
    ss >> hash;
    if (ss.fail()) {
      throw error("Neither the poplar executable nor the popart metadata "
                  "contains a hash number.");
    }

    return hash;
  }
};

Reader::Reader(const std::istream &in)
    : _impl(std::make_unique<ReaderImpl>(in)) {}
Reader::~Reader() = default;

size_t Reader::readExecutableHash() { return _impl->hash; }

bool Reader::containsPoplarExecutable() {
  return _impl->poplarExecutable.has_value();
}

bool Reader::containsExecutable() { return _impl->popartMetadata.has_value(); }

poplar::Executable Reader::deserializePoplarExecutable() {
  if (!containsPoplarExecutable()) {
    throw error("The file does not contain poplar executable.");
  }

  const popef::ExecutableReader &exeReader = _impl->poplarExecutable->get();
  return poplar::Executable::deserialize(
      std::move(exeReader.getStandaloneExecutableStream()));
}

std::unique_ptr<popart::popx::Executablex>
Reader::deserializeExecutable(popart::Ir &ir,
                              popart::popx::IrLowering &lowering) {
  if (!containsExecutable()) {
    throw error("The file does not contain popart metadata.");
  }

  const popef::OpaqueReader &metadataReader = _impl->popartMetadata->get();
  std::unique_ptr<std::istream> opaque_stream(
      std::move(metadataReader.getStandaloneDataStream()));
  auto popartMetadata = deserializePopartMetadata(*opaque_stream, ir, lowering);

  return popartMetadata;
}

} // namespace serialization
} // namespace popx
} // namespace popart
