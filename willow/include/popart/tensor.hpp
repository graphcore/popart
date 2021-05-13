// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef NEURALNET_TENSOR_HPP
#define NEURALNET_TENSOR_HPP

#include <map>
#include <memory>
#include <set>

#include <popart/error.hpp>
#include <popart/istepio.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensordebuginfo.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/vertex.hpp>
#include <popart/voiddata.hpp>

namespace popart {

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

class TensorTypeInfo {
public:
  TensorTypeInfo(TensorType, std::string);
  TensorType type() const;
  const std::string &type_s() const;

private:
  TensorType tensorType_;
  std::string tensor_type_;
};
const std::map<TensorType, TensorTypeInfo> &getTensorTypeInfoMap();
std::map<TensorType, TensorTypeInfo> initTensorTypeInfoMap();

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
  // The type of replication to use if a Stream tensor
  enum class ReplicatedStreamMode { Replicate, Broadcast };

  // note : producer (if there is one)
  // must be set after construction
  Tensor(TensorId, TensorType, Graph &, const DebugContext & = {});
  TensorId id;
  std::string str() const final { return id; }

  // a copy of this, but with no consumers or producer
  virtual std::unique_ptr<Tensor> clone(Graph &graph_) const;

  // ActGrad, Variable, etc:
  TensorType tensorType() const;
  const std::string &tensor_type() const;
  const TensorTypeInfo *getTensorTypeInfo() const { return tensorTypeInfo; }
  void setTensorType(TensorType);

  // Accessor's for the replicated stream mode
  ReplicatedStreamMode getReplicatedStreamMode() const {
    return replicatedStreamMode;
  }
  void setReplicatedStreamMode(const ReplicatedStreamMode &mode) {
    replicatedStreamMode = mode;
  }

  // Return all the pipeline stages the tensor is used in.
  std::set<PipelineStage> getPipelineStages() const;

  Consumers consumers;
  // shape and data type. Not to be used before inferShape of pir has run
  TensorInfo info;

  // information about tensor cached status
  TensorLocationInfo tensorLocationInfo;

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
   * Is this tensor produced by a hostLoad op?
   *
   * \returns true If producer is a hostLoad Op
   * \returns false Otherwise.
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

  template <typename... Args> void setTensorData(Args &&... args) {
    // if data has already been created and had a stream
    // connected to it, changing the data will lead to
    // the stream reading from the wrong address.
    // Rather use TensorData::resetData.
    if (data_) {
      throw error("attempting to setTensorData a second time");
    }
    data_.reset(new TensorData(std::forward<Args>(args)...));
  }

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
  VGraphIdAndTileSet
  getVirtualGraphIdAndTileSet(std::set<OpId> visited = {}) const;
  // Return the virtual graph id, or {-1, false} if there is not one
  VGraphIdAndTileSet
  getVirtualGraphIdAndTileSetUnsafe(std::set<OpId> visited = {}) const;

  // Determine the batch axis for this Tensor
  int getBatchAxis() const;

  bool consumersAllPreLoss() const;

  // Any of the consumers modify this tensor
  bool isModified() const;

  // Any of the consumers alias this tensor
  bool isAliased() const;

  // All regions modified by any of the Ops specified.
  // The Ops are tested in order, and the evaluation stops once the whole
  // tensor has been modified, or if all Ops have been tested
  view::Regions modifiedRegionsByOps(std::vector<Op *> ops) const;
  view::Regions modifiedRegionsByOps(std::vector<OpId> opIds) const;

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
  const TensorTypeInfo *tensorTypeInfo;

  // By default stream tensors are replicated
  ReplicatedStreamMode replicatedStreamMode = ReplicatedStreamMode::Replicate;

  // c++ note : we cannot initialise this as {nullptr} with gcc
  // when using pimpl, it must be initialised in the .cpp constructor
  std::unique_ptr<TensorData> data_;

  int getBatchAxisFromOp(Op *, bool, int) const;

  const TensorDebugInfo di;
};

class VariableTensor : public Tensor {
public:
  VariableTensor(TensorId, Graph &, const DebugContext & = {});

  std::unique_ptr<Tensor> clone(Graph &graph_) const override;

  void setVariableUpdateType(VariableUpdateType type) {
    variableUpdateType = type;
  }
  VariableUpdateType getVariableUpdateType() const {
    return variableUpdateType;
  }

  void setCopyFromTensor(TensorId value) { copyFromTensor = value; }
  TensorId getCopyFromTensor() { return copyFromTensor; }

private:
  VariableUpdateType variableUpdateType;

  // If the type is copy, this will identity where to copy from
  TensorId copyFromTensor;
};

struct PTensorCmp {
  bool operator()(Tensor *const &a, Tensor *const &b) const {
    return a->id < b->id;
  }
};

} // namespace popart

#endif
