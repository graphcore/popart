// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <fstream>

#include <boost/filesystem.hpp>

#include <popart/popx/executablex.hpp>
#include <popart/popx/executablexserialization.hpp>
#include <popart/popx/irlowering.hpp>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/optimizer.hpp>

#include <onnx/onnx_pb.h>

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
    std::map<TensorId, gcl::CollectiveBalancedHostRearrangement> &&cbrMap)
    : ir_lowering(ir_lowering_), deserialized(true),
      dataFlow(ir_lowering.ir().getDataFlow()),
      options(ir_lowering.ir().getSessionOptions()),
      executionMode(ir_lowering.ir().getExecutionMode()),
      tensors(std::move(tensorMap)), cbrHostRearrangement(std::move(cbrMap)) {
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
    std::map<TensorId, gcl::CollectiveBalancedHostRearrangement> &&cbrMap) {

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
  auto type         = lowering().getDeviceInfo()->getType();
  bool isHwCompatibleDevice =
      type == DeviceType::Ipu || type == DeviceType::OfflineIpu;

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
    if (tensor->info != TensorInfo(initializer)) {
      throw runtime_error("trying to reset weights using tensor with non "
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

const std::map<TensorId, gcl::CollectiveBalancedHostRearrangement>
Executablex::getCollectiveBalancedHostRearrangements() const {
  if (!deserialized) {
    std::map<TensorId, gcl::CollectiveBalancedHostRearrangement>
        hostRearrangements;
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

std::string
Executablex::getExecutablexCachePath(const std::string &cacheDir) const {
  size_t hash = ir().getHash();
  return logging::format("{}/{}.popart", cacheDir, hash);
}

void Executablex::serialize(const poplar::Executable &poplarExecutable,
                            const std::string &path) {
  // If target directory does not exist, create it
  auto target = boost::filesystem::path(path);
  if (target.has_parent_path()) {
    auto targetDir = target.parent_path();
    if (!boost::filesystem::exists(targetDir)) {
      logging::devicex::warn("Specified directory not found. "
                             "Creating {} directory ",
                             targetDir);
      if (!boost::filesystem::create_directories(targetDir))
        throw error("Cannot create cache directory. Aborting.");
    }
  }
  std::string filename = path;
  if (boost::filesystem::is_directory(target)) {
    filename = logging::format("{}/executable.popart", filename);
    logging::devicex::warn(
        "{} is a directory, saving serialized Executablex to {}",
        target.string(),
        filename);
  } else {
    logging::devicex::info("Saving serialized Executablex to {}", filename);
  }
  std::ofstream out(filename, std::ofstream::binary);
  if (!out.is_open()) {
    throw error("Unable to open file '{}'", filename);
  }
  serialize(poplarExecutable, out);
}

void Executablex::serialize(const poplar::Executable &poplarExecutable,
                            std::ostream &out) {
  popx::serialization::serializeExecutable(
      out, &poplarExecutable, this, ir().getHash());
}

poplar::Executable Executablex::getPoplarExecutable() {
  auto exe = lowering().getExecutable();

  if (!deserialized) {
    if (shouldSerialize()) {
      const std::string cachePath = ir().getSessionOptions().cachePath;
      serialize(exe, getExecutablexCachePath(cachePath));
    }
  }

  return exe;
}

} // namespace popx
} // namespace popart
