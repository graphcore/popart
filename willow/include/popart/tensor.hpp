// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TENSOR_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TENSOR_HPP_

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/dataflow.hpp>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/replicatedstreammode.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensordebuginfo.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/variablesettings.hpp>
#include <popart/vertex.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/debugcontext.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
class Aliases;
class Graph;
class Ir;
class Op;
class Tensor;

enum class TensorType {
  ActGrad = 0, // an activation or a gradient, basically any output of an Op
  Const,
  Stream,
  Unknown,
  Variable,
  N // number of tensor types
};

std::ostream &operator<<(std::ostream &os, const TensorType &tt);

// Define how the variable tensor will be updated
enum class VariableUpdateType {
  None = 0, // Not updated
  Gradient, // Is updated using it's gradient
  Copy      // Is updated by copying another tensor
};

// The consumers (Ops) of a Tensor. Note that
// one Op may consume a Tensor at multiple locations.
class Consumers {
public:
  // Consumers is specific to a unique Tensor, which is stored
  // for later use in the constructor
  Consumers(Tensor *tensorConsumed_);
  // The number of times an Op consumes a Tensor,
  // returns a non-negative integer
  int n(Op *) const;
  // increment the number of times an Op consumes
  void increment(Op *);
  // decrement the number of times an Op consumes
  void decrement(Op *);
  // increment the current counts with those in this map
  void extend(const std::map<Op *, int, POpCmp> &);
  // return the total number of consumptions, taking
  // into account Ops which consume multiple times,
  // so the sum over consuming nodes of the number of
  // times consumed
  int getTotal() const;
  // the number of times each consumer uses the Tensor
  const std::map<Op *, int, POpCmp> &getMap() const;
  // the pointers to the consumers, no duplication for
  // Ops which consume multiple times
  std::vector<Op *> getOps() const;

  // append information about this object
  void append(std::stringstream &ss);

  std::set<PipelineStage> getPipelineStages() const;
  std::set<VGraphId> getVirtualGraphIds() const;
  OptionalPipelineStage findLowestPipelineStage() const;
  OptionalPipelineStage findHighestPipelineStage() const;
  OptionalVGraphId findLowestVirtualGraphID() const;

private:
  // The number of times an Op consumes the Tensor which
  // owns these Consumers
  std::map<Op *, int, POpCmp> consumers_m;
  Tensor *tensorConsumed;
};

class TensorLocationInfo {
public:
  void setRemote(bool remote_) { remote = remote_; }
  bool isRemote() const { return remote; }

  void setSharded(bool sharded_) { sharded = sharded_; }

  bool isSharded() const { return sharded; }

  void setRemoteBufferInfo(RemoteBufferId rbId, RemoteBufferIndex index) {
    remoteBufferInfo = {rbId, index};
  }

  const std::pair<RemoteBufferId, RemoteBufferIndex>
  getRemoteBufferInfo() const {
    return remoteBufferInfo;
  }

  bool operator==(const TensorLocationInfo &rhs) const {
    return (remote == rhs.remote) && (sharded == rhs.sharded) &&
           (remoteBufferInfo == rhs.remoteBufferInfo);
  }

private:
  bool remote{false};
  bool sharded{false};
  std::pair<RemoteBufferId, RemoteBufferIndex> remoteBufferInfo;
};

class Tensor : public Vertex {
public:
  // note : producer (if there is one)
  // must be set after construction
  Tensor(TensorId, TensorType, Graph &, const DebugContext & = {});
  Tensor(TensorId, VariableSettings, Graph &, const DebugContext & = {});
  Tensor(TensorId,
         TensorType,
         VariableSettings,
         Graph &,
         const DebugContext & = {});
  TensorId id;
  std::string str() const final { return id; }

  // a copy of this, but with no consumers or producer
  virtual std::unique_ptr<Tensor> clone(Graph &graph_) const;

  // ActGrad, Variable, etc:
  TensorType tensorType() const;
  std::string tensor_type() const;
  void setTensorType(TensorType);

  // Accessor's for the replicated stream mode
  ReplicatedStreamMode getReplicatedStreamMode() const {
    return inputSettings.replicatedStreamMode();
  }
  void setReplicatedStreamMode(const ReplicatedStreamMode &mode) {
    inputSettings.setReplicatedStreamMode(mode);
  }

  void setTensorLocationInfo(
      TensorLocation &,
      std::pair<RemoteBufferId, RemoteBufferIndex> &remoteBufferInfo);

  // Return all the pipeline stages the tensor is used in.
  std::set<PipelineStage> getPipelineStages() const;

  Consumers consumers;
  // shape and data type. Not to be used before inferShape of pir has run
  TensorInfo info;

  // information about tensor cached status
  TensorLocationInfo tensorLocationInfo;

  // information about a (stream) tensor's input settings
  InputSettings inputSettings;

  // Similar to getProducer, but the user must handle the nullptr
  Op *getProducerUnsafe() const;
  Op *getProducer() const;
  void setProducer(Op *);
  void resetProducer(Op *);
  bool hasProducer() const;
  bool isGraphInput() const;
  InIndex getGraphInputIndex() const;
  bool isGraphOutput() const;
  OutIndex getGraphOutputIndex() const;
  bool isLoopInput() const;
  bool isImplicitLoopInput() const;
  bool isExplicitLoopInput() const;
  bool isLoopTripCounter() const;
  // Returns true if the tensor is not to be modified by an inplaced operation
  bool isUnmodifiable() const;
  // Returns true if the tensor is consumed by an implicit recompute operation
  // (which means the tensor is consumed implicitly and must not be modified)
  bool isCheckpointTensor() const;
  // Returns true if the tensor is produced by an implicit recompute operation
  bool isImplicitRecomputeTensor() const;
  // Returns true if the tensor is the target of a restore inplace
  bool isRestoreInplaceTensor() const;
  bool idIncludesPrefix(const std::vector<std::string> &) const;
  // Returns true for stream tensors that are optimizer tensors, as
  // well as their copies
  bool isOptimizerTensor() const;
  bool isRemoteArgTensor() const;
  bool isRandomSeedTensor() const;
  bool isOptimizerStateTensor() const;
  bool isAccumulatorTensor() const;
  /**
   * Is this tensor produced by a HostLoad Op or MultiExchangeOp with HostLoad
   * descriptor?
   *
   * \return true if producer is a HostLoad Op or MultiExchangeOp with HostLoad
   * descriptor false otherwise.
   */
  bool isHostLoadTensor() const;
  // Returns true for tensors that are weights (variables),
  // but not optimizer states or accumulators
  bool isWeightTensor() const;
  bool isAnchored() const;
  bool isRootAnchor() const;
  bool hasTensorData() const;
  TensorData *tensorData();
  const TensorData *tensorData() const;

  // Returns true if the tensor or any of it's aliases fulfill the predicate
  bool anyAlias(std::function<bool(Tensor *)> predicate) const;

  // Returns true if the tensor or any of it's aliases fulfill the predicate in
  // the given poprithm memory graph
  bool anyAliasFor(std::function<bool(Tensor *)> predicate,
                   const AliasModel &popMem) const;

  void setTensorDataFromCopyOf(const void *src, std::size_t size);
  void setTensorDataFromViewOf(void *src, std::size_t size);
  void setTensorDataByEmplaceOf(std::vector<char> &&data);
  // For copy/move ctors.
  // Note, we don't use universal forwarding so impl can be in .cpp, as that
  // requires a template, and we need impl to be in .cpp so we can use
  // std::make_unique, as header must be C++11.
  void setTensorData(const TensorData &td);
  void setTensorData(TensorData &&td);

  // Get all consumer ops and the producer op
  std::vector<Op *> associatedOps() const;

  Graph &getGraph() { return graph; }
  const Graph &getGraph() const { return graph; }

  Ir &getIr();
  const Ir &getIr() const;

  // Determine the virtual graph of this Tensor, on-the-fly
  // based on consumers and producers
  bool hasVirtualGraphId() const;

  VGraphId getVirtualGraphId() const;
  // return the virtual graph id, or -1 if there is not one
  VGraphId getVirtualGraphIdUnsafe() const;

  // Return the virtual graph id and io tile flag
  VGraphIdAndTileSet getVirtualGraphIdAndTileSet(std::set<OpId> &visited) const;
  // Return the virtual graph id, or {-1, false} if there is not one
  VGraphIdAndTileSet getVirtualGraphIdAndTileSetUnsafe() const;
  VGraphIdAndTileSet
  getVirtualGraphIdAndTileSetUnsafe(std::set<OpId> &visited) const;

  // Determine the batch axis for this Tensor
  int getBatchAxis() const;

  bool consumersAllPreLoss() const;

  /**
   * Check if any of the consumers modify this tensor
   * \param considerLoopInput If explicit loop inputs should be considered
   *                          as being modified. If false, only operations
   *                          modifying the tensor inplace will be considered.
   * \return                  True if the tensor is modified, otherwise false.
   */
  bool isModified(bool considerLoopInput = true) const;

  /**
   * Check if any of the consumers alias this tensor
   * \return       True if the tensor is aliased to any output, otherwise false.
   */
  bool isAliased() const;

  // All regions modified by any of the Ops specified.
  // The Ops are tested in order, and the evaluation stops once the whole
  // tensor has been modified, or if all Ops have been tested
  //
  // TODO T40061: Replace use of chain-based aliasing.
  view::Regions modifiedRegionsByOps(std::vector<Op *> ops,
                                     Aliases &aliases) const;
  view::Regions modifiedRegionsByOps(std::vector<OpId> opIds,
                                     Aliases &aliases) const;

  /**
   * Find operations that modify a tensor
   * \return All operations that (direct and indirectly) modify this tensor
   */
  std::set<Op *, POpCmp> getInplaceModifiers() const;

  /**
   * Find operations that modify a tensor with the given poprithm graph
   * \return All operations that (direct and indirectly) modify this tensor
   */
  std::set<Op *, POpCmp> getInplaceModifiersFor(const AliasModel *popMem) const;

  // Backtrack through input and parent graph tensors in order to get data from
  // initializer tensors (if they exist).
  // When ops are performed on initializers (e.g. slice), the
  // data is (intentionally) not inherited by the output tensors, this method
  // finds the original data and sets the data of the callee tensor.
  std::vector<char> getDataViaGraphTraversal() const;

  const popart::DebugInfo &getDebugInfo() const { return di; }

protected:
  Graph &graph;
  Op *producer;
  TensorType tensorType_;

  // c++ note : we cannot initialise this as {nullptr} with gcc
  // when using pimpl, it must be initialised in the .cpp constructor
  std::unique_ptr<TensorData> data_;

  int getBatchAxisFromOp(Op *, bool, int) const;

  bool anyAliasImpl(std::function<bool(Tensor *)> predicate,
                    const AliasModel &popMem,
                    const char *scopeDesc) const;

  const TensorDebugInfo di;

  /**
   * Members of old subclass VariableTensor
   * class VariableTensor : public Tensor {
   */
public:
  void setVariableUpdateType(VariableUpdateType type) {
    variableUpdateType = type;
  }
  VariableUpdateType getVariableUpdateType() const {
    return variableUpdateType;
  }

  void setCopyFromTensor(TensorId value) { copyFromTensor = value; }
  TensorId getCopyFromTensor() { return copyFromTensor; }

  /// \return The VariableSettings of this Variable
  VariableSettings getVariableSettings() const { return variableSettings; }

  /**
   * Returns the shape necessitated by IO.
   * \param replicationFactor The replication factor
   * \return the shape of the tensor, considering replica groups
   */
  std::vector<int64_t> returnedShape(unsigned replicationFactor);

  /**
   * Check that the info of a mutableVoidData object matches the expectations
   * set by the TensorInfo and VariableSettings. Throws an error if there is a
   * mismatch.
   * \param mutableVoidInfo The data of the MutableVoidInfo with the
   *                        same id as this tensor
   * \param replicationFactor The replicationFactor of
   *                          this instance
   */
  void verifyMutableVoidInfo(const TensorInfo mutableVoidInfo,
                             unsigned replicationFactor);

  /// Set the preparedVGraphIdAndTileSet
  void setPreparedVGraphIdAndTileSet();

private:
  VariableUpdateType variableUpdateType;
  VariableSettings variableSettings = VariableSettings();

  // If the type is copy, this will identity where to copy from
  TensorId copyFromTensor;

  // The virtual graph id and tile set after Ir::setIsPrepared is called
  VGraphIdAndTileSet preparedVGraphIdAndTileSet;
};

// Map and set classes for `popart::Tensor *` for deterministic iteration order.
template <typename T> using TensorMap = std::map<Tensor *, T, PTensorCmp>;
template <typename T>
using ConstTensorMap = std::map<const Tensor *, T, PTensorCmp>;
using TensorSet      = std::set<Tensor *, PTensorCmp>;
using ConstTensorSet = std::set<const Tensor *, PTensorCmp>;

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TENSOR_HPP_
