// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPEXECUTABLE_HPP
#define GUARD_NEURALNET_POPEXECUTABLE_HPP

#include <gcl/CollectiveBalancedReorder.hpp>

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

namespace ONNX_NAMESPACE {
class ModelProto;
}

namespace popart {
class StepIOSplitter;
class Optimizer;
namespace popx {
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
  nonstd::optional<std::map<TensorId, gcl::CollectiveBalancedHostRearrangement>>
      cbrHostRearrangement;

public:
  Executablex(IrLowering &ir_lowering_);
  Executablex(
      IrLowering &ir_lowering_,
      std::unordered_map<TensorId, std::unique_ptr<Tensor>> &&tensorMap,
      std::map<TensorId, gcl::CollectiveBalancedHostRearrangement> &&cbrMap);

  static std::unique_ptr<Executablex>
  createFromLoweredIr(IrLowering &ir_lowering_);

  static std::unique_ptr<Executablex> createFromStream(
      IrLowering &ir_lowering_,
      std::unordered_map<TensorId, std::unique_ptr<Tensor>> &&tensorMap,
      std::map<TensorId, gcl::CollectiveBalancedHostRearrangement> &&cbrMap);

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

  const std::map<TensorId, gcl::CollectiveBalancedHostRearrangement>
  getCollectiveBalancedHostRearrangements() const;

  static std::string getExecutablexCachePath(const std::string &cachePath);

  // Serialize this object and save the ouptut to disk.
  void saveExecutablex();

  void updateOptimizerTensors();

  // Get the poplar::Executable from the IrLowering. If the executable
  // was cached on disk then the cached executable will be returned.
  // If not the graph is compiled and the resulting executable is returned.
  // If engine caching is enabled then the graph compilation result is
  // serialized and stored to disk. In this case the `popx::Executablex' is
  // also serialized and stored to `ir().getSessionOptions().cachePath'.
  // The logic has been kept this way to
  poplar::Executable getPoplarExecutable();
};

} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_POPEXECUTABLE_HPP
