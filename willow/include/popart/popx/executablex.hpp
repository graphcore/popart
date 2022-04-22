// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPEXECUTABLE_HPP
#define GUARD_NEURALNET_POPEXECUTABLE_HPP

#include <cstdint>
#include <gcl/CollectiveBalancedReorder.hpp>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"

namespace ONNX_NAMESPACE {
class ModelProto;
}

namespace popart {
class DataFlow;
struct SessionOptions;

namespace popx {
class IrLowering;

class Executablex {
private:
  IrLowering &ir_lowering;
  bool deserialized = false;

  // These structures are references to structures stored or deserialized in the
  // IR.
  const DataFlow &dataFlow;
  const SessionOptions &options;
  const Ir::ExecutionMode executionMode;

  // These Tensor are handles to Tensors stored in the IR(graphs).
  // If the executable is deserialized they are handles to the tensors mapping
  // since we don't fully reconstruct the IR when deserializing.
  std::vector<Tensor *> weightTensors;
  std::vector<Tensor *> anchorTensors;
  std::vector<Tensor *> optimizerTensors;
  std::vector<Tensor *> dataStreamTensors;
  Tensor *seedTensor = nullptr;

  std::vector<TensorId> getTensorIds(TensorType);

  // We only populate these structures during deserialization to
  // avoid unneccessary copies
  nonstd::optional<std::unordered_map<TensorId, std::unique_ptr<Tensor>>>
      tensors;
  nonstd::optional<std::map<TensorId, CollectiveBalancedReorderId>>
      cbrHostRearrangementIds;
  nonstd::optional<std::map<CollectiveBalancedReorderId,
                            gcl::CollectiveBalancedHostRearrangement>>
      cbrHostRearrangements;

public:
  Executablex(IrLowering &ir_lowering_);
  Executablex(IrLowering &ir_lowering_,
              std::unordered_map<TensorId, std::unique_ptr<Tensor>> &&tensorMap,
              std::map<TensorId, CollectiveBalancedReorderId> &&cbrIdMap,
              std::map<CollectiveBalancedReorderId,
                       gcl::CollectiveBalancedHostRearrangement> &&cbrMap);

  static std::unique_ptr<Executablex>
  createFromLoweredIr(IrLowering &ir_lowering_);

  static std::unique_ptr<Executablex> createFromStream(
      IrLowering &ir_lowering_,
      std::unordered_map<TensorId, std::unique_ptr<Tensor>> &&tensorMap,
      std::map<TensorId, CollectiveBalancedReorderId> &&cbrIdMap,
      std::map<CollectiveBalancedReorderId,
               gcl::CollectiveBalancedHostRearrangement> &&cbrMap);

  IrLowering &lowering();
  const IrLowering &lowering() const;

  const Ir &ir() const;

  bool isDeserialized() const { return deserialized; }
  bool shouldSerialize();

  bool containsTensor(const TensorId &id) const;
  Tensor *getTensor(const TensorId &);
  const Tensor *getTensor(const TensorId &) const;

  void setRandomSeedValue(uint64_t value);
  void resetWeights(
      const ONNX_NAMESPACE::ModelProto &modelProto,
      const bool ignoreWeightsInModelWithoutCorrespondingIrWeight = false);

  const SessionOptions &getSessionOptions() const { return options; }

  const std::vector<Tensor *> &getWeightTensors() const {
    return weightTensors;
  }

  const std::vector<Tensor *> &getAnchorTensors() const {
    return anchorTensors;
  }

  const std::vector<Tensor *> &getOptimizerTensors() const {
    return optimizerTensors;
  }

  const std::vector<Tensor *> &getDataStreamTensors() const {
    return dataStreamTensors;
  }

  const Tensor *getSeedTensor() const { return seedTensor; }

  const gcl::CollectiveBalancedHostRearrangement &
  getCollectiveBalancedHostRearrangement(const TensorId &id) const;

  const std::map<CollectiveBalancedReorderId,
                 gcl::CollectiveBalancedHostRearrangement>
  getCollectiveBalancedHostRearrangements() const;

  const std::map<TensorId, CollectiveBalancedReorderId>
  getCollectiveBalancedHostRearrangementIds() const;

  std::string getCachePath(const std::string &cacheDir) const;

  void updateOptimizerTensors();
};

} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_POPEXECUTABLE_HPP
