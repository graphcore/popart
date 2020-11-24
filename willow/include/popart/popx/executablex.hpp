// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPEXECUTABLE_HPP
#define GUARD_NEURALNET_POPEXECUTABLE_HPP

#include <popart/devicemanager.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/popprograms.hpp>
#include <popart/popx/poptensors.hpp>
#include <popart/popx/pritask.hpp>
#include <popart/vendored/optional.hpp>

#include <set>
#include <tuple>
#include <popart/names.hpp>

#include <popart/ir.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {
class StepIOSplitter;

namespace popx {
class CollectiveBalancedHostRearrangement;
class Devicex;

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
  nonstd::optional<std::map<TensorId, CollectiveBalancedHostRearrangement>>
      cbrHostRearrangement;

public:
  Executablex(IrLowering &ir_lowering_);
  Executablex(IrLowering &ir_lowering_,
              std::unordered_map<TensorId, std::unique_ptr<Tensor>> tensorMap,
              std::map<TensorId, CollectiveBalancedHostRearrangement> &&cbrMap);

  static std::unique_ptr<Executablex>
  createFromLoweredIr(IrLowering &ir_lowering_);

  static std::unique_ptr<Executablex> createFromStream(
      IrLowering &ir_lowering_,
      std::unordered_map<TensorId, std::unique_ptr<Tensor>> tensorMap,
      std::map<TensorId, CollectiveBalancedHostRearrangement> &&cbrMap);

  IrLowering &lowering();
  const IrLowering &lowering() const;

  const Ir &ir() const;

  Tensor *getTensor(const TensorId &);
  const Tensor *getTensor(const TensorId &) const;

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

  const CollectiveBalancedHostRearrangement &
  getCollectiveBalancedHostRearrangement(const TensorId &id) const;

  const std::map<TensorId, CollectiveBalancedHostRearrangement>
  getCollectiveBalancedHostRearrangements() const;
};

} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_POPEXECUTABLE_HPP
