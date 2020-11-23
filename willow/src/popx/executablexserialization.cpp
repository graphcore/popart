// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/intervals.hpp>
#include <popart/ir.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/optimizer.hpp>
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

popart::cap::AnchorReturnTypeId toCapnpArtId(popart::AnchorReturnTypeId id) {
  switch (id) {
  case popart::AnchorReturnTypeId::Final:
    return popart::cap::AnchorReturnTypeId::FINAL;
  case popart::AnchorReturnTypeId::EveryN:
    return popart::cap::AnchorReturnTypeId::EVERY_N;
  case popart::AnchorReturnTypeId::All:
    return popart::cap::AnchorReturnTypeId::ALL;
  case popart::AnchorReturnTypeId::Sum:
    return popart::cap::AnchorReturnTypeId::SUM;
  }

  throw error("Invalid AnchorReturnTypeId {}", id);
}

popart::cap::SyntheticDataMode
toCapnpSyntheticDataMode(popart::SyntheticDataMode mode) {
  switch (mode) {
  case popart::SyntheticDataMode::Off:
    return popart::cap::SyntheticDataMode::OFF;
  case popart::SyntheticDataMode::Zeros:
    return popart::cap::SyntheticDataMode::ZEROS;
  case popart::SyntheticDataMode::RandomNormal:
    return popart::cap::SyntheticDataMode::RANDOM_NORMAL;
  case popart::SyntheticDataMode::N:
    return popart::cap::SyntheticDataMode::N;
  }

  throw error("Invalid SyntheticDataMode {}", static_cast<int>(mode));
}

popart::cap::TensorType toCapnpTensorType(popart::TensorType type) {
  switch (type) {
  case popart::TensorType::ActGrad:
    return popart::cap::TensorType::ACT_GRAD;
  case popart::TensorType::Const:
    return popart::cap::TensorType::CONSTANT;
  case popart::TensorType::Momentum:
    return popart::cap::TensorType::MOMENTUM;
  case popart::TensorType::Stream:
    return popart::cap::TensorType::STREAM;
  case popart::TensorType::Unknown:
    return popart::cap::TensorType::UNKNOWN;
  case popart::TensorType::Variable:
    return popart::cap::TensorType::VARIABLE;
  case popart::TensorType::Cache:
    return popart::cap::TensorType::CACHE;
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
  case popart::cap::TensorType::MOMENTUM:
    return popart::TensorType::Momentum;
  case popart::cap::TensorType::STREAM:
    return popart::TensorType::Stream;
  case popart::cap::TensorType::UNKNOWN:
    return popart::TensorType::Unknown;
  case popart::cap::TensorType::VARIABLE:
    return popart::TensorType::Variable;
  case popart::cap::TensorType::CACHE:
    return popart::TensorType::Cache;
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

void serializeExecutable(std::ostream &out,
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
    auto tensorTileMap        = ir_lowering.getTensorTileMap();
    auto tensorTileMapBuilder = irLoweringBuilder.initTensorTileMap();
    auto capnpTensorTileMap =
        tensorTileMapBuilder.initMappings(tensorTileMap.size());

    int i = 0;
    for (const auto &kv : tensorTileMap) {
      capnpTensorTileMap[i].setId(kv.first);
      auto tensorIntervalListsBuilder =
          capnpTensorTileMap[i].initTensorIntervalLists(kv.second.size());

      const auto &tensorIntervalLists = kv.second;
      for (int j = 0; j < tensorIntervalLists.size(); ++j) {
        const auto &tensorIntervalList = tensorIntervalLists[j];
        auto innerTensorIntervalListsBuilder =
            tensorIntervalListsBuilder.init(j, tensorIntervalList.size());
        for (int k = 0; k < tensorIntervalList.size(); ++k) {

          innerTensorIntervalListsBuilder[k].setStart(
              tensorIntervalList[k].first);
          innerTensorIntervalListsBuilder[k].setEnd(
              tensorIntervalList[k].second);
        }
      }
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
    auto hostReduceStreamIds = ir_lowering.getHostReduceStreamIds();
    auto hostReduceStreamIdsBuilder =
        irLoweringBuilder.initHostReduceStreamIds(hostReduceStreamIds.size());
    int i = 0;
    for (const auto &tid : hostReduceStreamIds) {
      hostReduceStreamIdsBuilder.set(i, tid);
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
    auto anchorTensors     = ir.getDataFlow().anchors();
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
      Tensor *tensor     = ir.getTensor(id);
      auto tensorBuilder = tensors[i];
      serializeTensor(tensor, tensorBuilder);
      ++i;
    }

    for (auto &id : ir.getDataFlow().anchors()) {
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
    const auto &collectiveBalancedReorders =
        ir_lowering.getCollectiveReorders();

    auto hostRearrangementsBuilder =
        executablexBuilder.initCollectiveBalancedHostRearrangements();
    auto rearrangementsBuilder = hostRearrangementsBuilder.initRearrangements(
        collectiveBalancedReorders.size());

    int i = 0;
    for (const auto &kv : collectiveBalancedReorders) {
      rearrangementsBuilder[i].setId(kv.first);

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
deserializeExecutable(std::istream &in,
                      popart::Ir &ir,
                      popart::popx::IrLowering &lowering) {
  kj::std::StdInputStream sis(in);
  capnp::InputStreamMessageReader message(sis);

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
    auto capnpTensorTileMap = irLoweringReader.getTensorTileMap().getMappings();
    TensorTileMap mappings;
    for (const auto m : capnpTensorTileMap) {
      auto capnpTensorIntervalLists = m.getTensorIntervalLists();
      std::vector<TensorIntervalList> tils;
      tils.reserve(capnpTensorIntervalLists.size());
      for (const auto capnpTensorIntervalList : capnpTensorIntervalLists) {
        std::vector<TensorInterval> tensorIntervalList;
        tensorIntervalList.reserve(capnpTensorIntervalList.size());
        for (const auto ti : capnpTensorIntervalList) {
          tensorIntervalList.push_back(
              std::make_pair(ti.getStart(), ti.getEnd()));
        }
        tils.emplace_back(std::move(tensorIntervalList));
      }
      std::string id = m.getId();
      mappings.emplace(id, std::move(tils));
    }
    lowering.setTensorTileMap(mappings);
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
    auto hostReduceStreamIds = irLoweringReader.getHostReduceStreamIds();
    std::vector<TensorId> hostReduceStreamIds_;
    hostReduceStreamIds_.reserve(hostReduceStreamIds.size());

    for (const auto t : hostReduceStreamIds) {
      hostReduceStreamIds_.push_back(t);
    }
    lowering.getHostReduceStreamIds() = hostReduceStreamIds_;
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

    auto gid = popart::GraphId("");
    popart::Graph dummyGraph(ir, gid);
    for (const auto capnpTensor : tensors) {
      std::string id = capnpTensor.getId();
      auto type      = capnpTensor.getTensorType();
      auto tensor    = std::make_unique<popart::Tensor>(
          id, toPopartTensorType(type), dummyGraph);

      auto capnpTensorInfo   = capnpTensor.getTensorInfo();
      auto capnpDataTypeInfo = capnpTensorInfo.getDataTypeInfo();
      popart::DataType dataType =
          toPopartDataType(capnpDataTypeInfo.getDataType());
      auto shapeReader = capnpTensorInfo.getShape();
      std::vector<int64_t> shape;
      for (const auto s : shapeReader) {
        shape.push_back(s);
      }

      tensor->info = popart::TensorInfo(dataType, shape);

      auto capnpTensorLocationInfo = capnpTensor.getTensorLocationInfo();
      tensor->tensorLocationInfo.setSharded(
          capnpTensorLocationInfo.getSharded());
      tensor->tensorLocationInfo.setRemote(capnpTensorLocationInfo.getRemote());
      tensor->tensorLocationInfo.setRemoteBufferInfo(
          capnpTensorLocationInfo.getRemoteBufferInfo().getId(),
          capnpTensorLocationInfo.getRemoteBufferInfo().getIndex());

      if (capnpTensor.hasTensorData()) {
        auto tensorDataReader = capnpTensor.getTensorData();
        const void *src       = tensorDataReader.begin();
        tensor->setTensorData(tensor->info, src);
      }

      deserializedTensors[id] = std::move(tensor);
    }
  }

  std::map<TensorId, CollectiveBalancedHostRearrangement> cbrHostRearrangement;
  {
    auto collectiveBalancedHostRearrangementsReader =
        executablexReader.getCollectiveBalancedHostRearrangements();
    auto rearrangementsReader =
        collectiveBalancedHostRearrangementsReader.getRearrangements();

    for (const auto cbr : rearrangementsReader) {
      std::string id           = cbr.getId();
      auto rearrangementReader = cbr.getRearrangement();

      CollectiveBalancedHostRearrangement cbhr;
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

      cbrHostRearrangement[id] = cbhr;
    }
  }

  return popart::popx::Executablex::createFromStream(
      lowering,
      std::move(deserializedTensors),
      std::move(cbrHostRearrangement));
}

} // namespace serialization
} // namespace popx
} // namespace popart
