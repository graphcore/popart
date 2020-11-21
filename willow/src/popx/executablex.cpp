// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <fstream>

#include <boost/filesystem.hpp>

#include <popart/popx/executablex.hpp>
#include <popart/popx/executablexserialization.hpp>
#include <popart/popx/irlowering.hpp>

#include <popart/ir.hpp>
#include <popart/op/getrandomseed.hpp>

namespace popart {
namespace popx {

Executablex::Executablex(IrLowering &ir_lowering_)
    : ir_lowering(ir_lowering_), deserialized(false),
      dataFlow(ir_lowering.ir().getDataFlow()),
      options(ir_lowering.ir().getSessionOptions()),
      executionMode(ir_lowering.ir().getExecutionMode()) {
  for (auto &id : ir().getTensorIds(TensorType::Variable)) {
    Tensor *tensor = ir().getTensor(id);
    weightTensors.push_back(tensor);
  }

  for (auto &id : ir().getDataFlow().anchors()) {
    Tensor *tensor = ir().getTensor(id);
    // anchorTensors[id] = tensor;
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
  }
}

Executablex::Executablex(
    IrLowering &ir_lowering_,
    std::unordered_map<TensorId, std::unique_ptr<Tensor>> tensorMap,
    std::map<TensorId, CollectiveBalancedHostRearrangement> &&cbrMap)
    : ir_lowering(ir_lowering_), deserialized(true),
      dataFlow(ir_lowering.ir().getDataFlow()),
      options(ir_lowering.ir().getSessionOptions()),
      executionMode(ir_lowering.ir().getExecutionMode()),
      tensors(std::move(tensorMap)), cbrHostRearrangement(std::move(cbrMap)) {
  auto weightTensorIds = getTensorIds(TensorType::Variable);
  weightTensors.reserve(weightTensorIds.size());
  for (auto &id : weightTensorIds) {
    Tensor *tensor = getTensor(id);
    weightTensors.push_back(tensor);
  }

  for (auto &id : ir().getDataFlow().anchors()) {
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
  }
}

std::unique_ptr<Executablex>
Executablex::createFromLoweredIr(IrLowering &ir_lowering_) {
  return std::make_unique<Executablex>(ir_lowering_);
}

std::unique_ptr<Executablex> Executablex::createFromStream(
    IrLowering &ir_lowering_,
    std::unordered_map<TensorId, std::unique_ptr<Tensor>> tensorMap,
    std::map<TensorId, CollectiveBalancedHostRearrangement> &&cbrMap) {

  return std::make_unique<Executablex>(
      ir_lowering_, std::move(tensorMap), std::move(cbrMap));
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
  auto cachePath    = ir().getSessionOptions().cachePath;
  auto cacheEnabled = ir().getSessionOptions().enableEngineCaching;

  const bool shouldSerialize =
      cacheEnabled && !cachePath.empty() &&
      lowering().getDeviceInfo()->getType() == DeviceType::Ipu && !deserialized;

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

std::vector<TensorId> Executablex::getTensorIds(TensorType type) {
  if (!tensors) {
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

const CollectiveBalancedHostRearrangement &
Executablex::getCollectiveBalancedHostRearrangement(const TensorId &id) const {
  if (!deserialized) {
    return lowering().getCollectiveBalancedHostRearrangement(id);
  }

  const auto &cbrHostRearrangement_ = cbrHostRearrangement.value();
  auto found                        = cbrHostRearrangement_.find(id);
  if (found == cbrHostRearrangement_.end()) {
    throw error("CollectiveBalancedReorderHostRearrangement does not exist for "
                "tensor {}",
                id);
  }
  return found->second;
}

const std::map<TensorId, CollectiveBalancedHostRearrangement>
Executablex::getCollectiveBalancedHostRearrangements() const {
  if (!deserialized) {
    std::map<TensorId, CollectiveBalancedHostRearrangement> hostRearrangements;
    const auto &cbrs = lowering().getCollectiveReorders();
    for (const auto &kv : cbrs) {
      const auto id                = kv.first;
      const auto hostRearrangement = getCollectiveBalancedHostRearrangement(id);
      hostRearrangements[id]       = hostRearrangement;
    }
    return hostRearrangements;
  }

  return cbrHostRearrangement.value();
}

std::string Executablex::getExecutablexCachePath(const std::string &cachePath) {
  return cachePath + ".popart.exe";
}

void Executablex::saveExecutablex() {
  if (false == shouldSerialize()) {
    logging::devicex::warn("Serialization is disabled. Skipping save.");
    return;
  }

  auto cachePath = ir().getSessionOptions().cachePath;

  // If target directory does not exist, create it
  auto cachePathObj = boost::filesystem::path(cachePath);
  if (cachePathObj.has_parent_path()) {
    auto cacheDir = cachePathObj.parent_path();
    if (!boost::filesystem::exists(cacheDir)) {
      logging::devicex::warn("Specified cache directory not found. "
                             "Creating {} directory ",
                             cacheDir);
      if (!boost::filesystem::create_directory(cacheDir))
        throw error("Cannot create cache directory. Aborting.");
    }
  }
  auto serializedExecutableFilePath = getExecutablexCachePath(cachePath);
  logging::devicex::info("Saving serialized Executablex to {}",
                         serializedExecutableFilePath);
  std::ofstream out(serializedExecutableFilePath);
  popx::serialization::serializeExecutable(out, *this);
}

poplar::Executable Executablex::getPoplarExecutable() {
  auto exe = lowering().getExecutable();

  if (!deserialized) {
    if (shouldSerialize()) {
      saveExecutablex();
      lowering().trySavePoplarExecutable(exe);
    }
  }

  return std::move(exe);
}

} // namespace popx
} // namespace popart
