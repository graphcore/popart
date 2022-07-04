// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gcl/CollectiveBalancedReorder.hpp>
#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/irlowering.hpp>

#include "popart/devicemanager.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/replicatedtensorshardingbundle.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordata.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensornames.hpp"
#include "popart/variablesettings.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
namespace popx {

Executablex::Executablex(IrLowering &ir_lowering_)
    : ir_lowering(ir_lowering_), deserialized(false),
      dataFlow(ir_lowering.ir().getDataFlow()),
      options(ir_lowering.ir().getSessionOptions()),
      executionMode(ir_lowering.ir().getExecutionMode()) {
  for (auto &id : ir().getTensorIds(TensorType::Variable)) {
    Tensor *tensor = ir().getTensor(id);
    if (!tensor->hasProducer()) {
      weightTensors.push_back(tensor);
    }
  }

  for (auto &id : ir().getRootAnchors()) {
    Tensor *tensor = ir().getTensor(id);
    anchorTensors.push_back(tensor);
  }

  for (auto *tensor : ir().optimizerTensors()) {
    optimizerTensors.push_back(tensor);
  }

  for (auto *tensor : ir().dataStreamTensors()) {
    dataStreamTensors.push_back(tensor);
  }

  if (ir().getRequiresRandomSeed()) {
    TensorId seedId = GetRandomSeedOp::getStreamedSeedTensorId();
    seedTensor      = ir().getTensor(seedId);
    uint64_t init = std::chrono::system_clock::now().time_since_epoch().count();
    setRandomSeedValue(init);
  }
}

Executablex::Executablex(
    IrLowering &ir_lowering_,
    std::unordered_map<TensorId, std::unique_ptr<Tensor>> &&tensorMap,
    std::map<TensorId, CollectiveBalancedReorderId> &&cbrIdMap,
    std::map<CollectiveBalancedReorderId,
             gcl::CollectiveBalancedHostRearrangement> &&cbrMap)
    : ir_lowering(ir_lowering_), deserialized(true),
      dataFlow(ir_lowering.ir().getDataFlow()),
      options(ir_lowering.ir().getSessionOptions()),
      executionMode(ir_lowering.ir().getExecutionMode()),
      tensors(std::move(tensorMap)),
      cbrHostRearrangementIds(std::move(cbrIdMap)),
      cbrHostRearrangements(std::move(cbrMap)) {
  auto weightTensorIds = getTensorIds(TensorType::Variable);
  weightTensors.reserve(weightTensorIds.size());
  for (auto &id : weightTensorIds) {
    Tensor *tensor = getTensor(id);
    if (!tensor->hasProducer()) {
      weightTensors.push_back(tensor);
    }
  }

  for (auto &id : ir().getRootAnchors()) {
    Tensor *tensor = getTensor(id);
    anchorTensors.push_back(tensor);
  }

  for (auto &id : getTensorIds(TensorType::Stream)) {
    Tensor *tensor = getTensor(id);
    if (tensor->isOptimizerTensor()) {
      optimizerTensors.push_back(tensor);
    } else if (!tensor->isRandomSeedTensor()) {
      dataStreamTensors.push_back(tensor);
    }
  }

  if (ir().getRequiresRandomSeed()) {
    TensorId seedId = GetRandomSeedOp::getStreamedSeedTensorId();
    seedTensor      = getTensor(seedId);
    // Don't initialize the seed tensor value here since it's deserialized
  }
}

std::unique_ptr<Executablex>
Executablex::createFromLoweredIr(IrLowering &ir_lowering_) {
  return std::make_unique<Executablex>(ir_lowering_);
}

std::unique_ptr<Executablex> Executablex::createFromStream(
    IrLowering &ir_lowering_,
    std::unordered_map<TensorId, std::unique_ptr<Tensor>> &&tensorMap,
    std::map<TensorId, CollectiveBalancedReorderId> &&cbrIdMap,
    std::map<CollectiveBalancedReorderId,
             gcl::CollectiveBalancedHostRearrangement> &&cbrMap) {

  return std::make_unique<Executablex>(ir_lowering_,
                                       std::move(tensorMap),
                                       std::move(cbrIdMap),
                                       std::move(cbrMap));
}

const IrLowering &Executablex::lowering() const { return ir_lowering; }

IrLowering &Executablex::lowering() { return ir_lowering; }

const popart::Ir &Executablex::ir() const { return ir_lowering.ir(); }

bool Executablex::containsTensor(const TensorId &id) const {
  if (!deserialized) {
    return ir().containsTensor(id);
  }

  const auto &tensors_ = tensors.value();
  return tensors_.count(id) > 0;
}

bool Executablex::shouldSerialize() {
  auto cachePath            = ir().getSessionOptions().cachePath;
  auto cacheEnabled         = ir().getSessionOptions().enableEngineCaching;
  auto isHwCompatibleDevice = lowering().getDeviceInfo()->canCompileOffline();

  const bool shouldSerialize = cacheEnabled && !cachePath.empty() &&
                               isHwCompatibleDevice && !deserialized;

  return shouldSerialize;
}

Tensor *Executablex::getTensor(const TensorId &id) {
  if (!deserialized) {
    return ir().getTensor(id);
  }

  const auto &tensors_ = tensors.value();
  auto found_iter      = tensors_.find(id);
  const bool found     = found_iter != tensors_.end();
  if (!found) {
    throw error("TensorId {} does not exist", id);
  }

  return found_iter->second.get();
}

const Tensor *Executablex::getTensor(const TensorId &id) const {
  if (!deserialized) {
    return ir().getTensor(id);
  }

  const auto &tensors_ = tensors.value();
  auto found_iter      = tensors_.find(id);
  const bool found     = found_iter != tensors_.end();
  if (!found) {
    throw error("TensorId {} does not exist", id);
  }

  return found_iter->second.get();
}

std::set<TensorId> Executablex::getAllTensorIds() {
  if (!deserialized) {
    return ir().getAllTensorIds();
  }

  const auto &tensors_ = tensors.value();

  std::set<TensorId> ids;
  for (auto &tensor : tensors_) {
    ids.insert(tensor.first);
  }

  return ids;
}

std::vector<TensorId> Executablex::getTensorIds(TensorType type) {
  if (!deserialized) {
    return ir().getTensorIds(type);
  }

  const auto &tensors_ = tensors.value();

  std::vector<TensorId> ids;
  for (auto &tensor : tensors_) {
    if (tensor.second->tensorType() == type) {
      ids.push_back(tensor.first);
    }
  }
  return ids;
}

void Executablex::setRandomSeedValue(uint64_t value) {
  logging::devicex::info("Setting the random seed to {}", value);
  TensorId seedId    = GetRandomSeedOp::getStreamedSeedTensorId();
  Tensor *seedTensor = getTensor(seedId);
  std::vector<char> seedData(seedTensor->info.nbytes());
  *reinterpret_cast<uint64_t *>(seedData.data()) = value;
  if (seedTensor->hasTensorData()) {
    seedTensor->tensorData()->resetData(seedTensor->info, seedData.data());
  } else {
    seedTensor->setTensorData(seedTensor->info, seedData.data());
  }
}

void Executablex::resetWeights(
    const ONNX_NAMESPACE::ModelProto &modelProto,
    const bool ignoreWeightsInModelWithoutCorrespondingIrWeight) {
  auto &onnxGraph = modelProto.graph();

  auto replicationFactor = 0;

  for (auto &m : modelProto.metadata_props()) {
    if (m.key() == sReplicationFactor) {
      replicationFactor = static_cast<size_t>(std::stoi(m.value()));
      break;
    }
  }

  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    if (!containsTensor(tenId)) {
      if (ignoreWeightsInModelWithoutCorrespondingIrWeight) {
        continue;
      } else {
        throw runtime_error("resetWeights, no tensor '" + tenId +
                            "' in tensors");
      }
    }
    auto tensor = getTensor(tenId);
    auto groupCount =
        tensor->getVariableSettings().getGroupCount(replicationFactor);

    if (replicationFactor > 0 && groupCount != 1) {
      auto info           = TensorInfo(initializer);
      Shape grouped_shape = Shape(0);
      grouped_shape.push_back(groupCount);
      grouped_shape.insert(grouped_shape.begin() + 1,
                           tensor->info.shape().begin(),
                           tensor->info.shape().end());
      if (grouped_shape != info.shape() ||
          tensor->info.dataType() != info.dataType()) {
        throw runtime_error("Trying to reset weights using tensor with non "
                            "matching tensor info. Tensor ID: {}",
                            tensor->id);
      }
    } else if (tensor->info != TensorInfo(initializer)) {
      throw runtime_error("Trying to reset weights using tensor with non "
                          "matching tensor info. Tensor ID: {}",
                          tensor->id);
    }
    tensor->tensorData()->resetData(initializer);
  }
}

void Executablex::updateOptimizerTensors() {
  for (auto *optTensor : optimizerTensors) {
    ir().getOptimizer().resetTensorData(*optTensor);
  }
}

const gcl::CollectiveBalancedHostRearrangement &
Executablex::getCollectiveBalancedHostRearrangement(const TensorId &id) const {
  if (!deserialized) {
    return lowering()
        .getReplicatedTensorShardingBundle()
        .getCollectiveBalancedHostRearrangement(id);
  }

  auto remoteArgId = getRemoteArgTensorId(stripAllReservedPrefixes(id));

  const auto &cbrHostRearrangementId_ = cbrHostRearrangementIds.value();
  const auto &cbrHostRearrangement_   = cbrHostRearrangements.value();
  auto found                          = cbrHostRearrangementId_.find(id);
  auto foundRemoteArg = cbrHostRearrangementId_.find(remoteArgId);
  if (found != cbrHostRearrangementId_.end()) {
    return cbrHostRearrangement_.at(found->second);
  } else if (foundRemoteArg != cbrHostRearrangementId_.end()) {
    return cbrHostRearrangement_.at(foundRemoteArg->second);
  } else {
    throw error("CollectiveBalancedReorderHostRearrangement does not exist for "
                "tensor {}",
                id);
  }
}

const std::map<CollectiveBalancedReorderId,
               gcl::CollectiveBalancedHostRearrangement>
Executablex::getCollectiveBalancedHostRearrangements() const {
  if (!deserialized) {
    std::map<CollectiveBalancedReorderId,
             gcl::CollectiveBalancedHostRearrangement>
        hostRearrangements;
    const auto &cbrs =
        lowering().getReplicatedTensorShardingBundle().getCollectiveReorders();
    for (const auto &kv : cbrs) {
      const auto cbrId             = kv.first;
      const auto hostRearrangement = kv.second->getHostRearrangement();
      hostRearrangements[cbrId]    = hostRearrangement;
    }
    return hostRearrangements;
  }

  return cbrHostRearrangements.value();
}

const std::map<TensorId, CollectiveBalancedReorderId>
Executablex::getCollectiveBalancedHostRearrangementIds() const {
  if (!deserialized) {
    const auto &cbrIds = lowering()
                             .getReplicatedTensorShardingBundle()
                             .getCollectiveReorderIds();
    return cbrIds;
  }

  return cbrHostRearrangementIds.value();
}

std::string Executablex::getCachePath(const std::string &cacheDir) const {
  size_t hash = ir().getHash();
  return logging::format("{}/{}.popef", cacheDir, hash);
}

} // namespace popx
} // namespace popart
