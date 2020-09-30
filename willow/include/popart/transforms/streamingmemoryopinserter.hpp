// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STREAMING_MEMORY_OP_INSERTER_HPP
#define GUARD_NEURALNET_STREAMING_MEMORY_OP_INSERTER_HPP

#include <iostream>
#include <map>

#include <popart/op.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class RemoteLoadOp;
class RemoteStoreOp;
class ReplicatedAllGatherOp;
class ReplicatedAllReduceOp;
class ReplicatedReduceScatterOp;

class StreamingMemoryOpInserter {
public:
  // Default constructor (don't allow).
  StreamingMemoryOpInserter() = delete;
  // Copy constructor (don't allow).
  StreamingMemoryOpInserter(const StreamingMemoryOpInserter &rhs) = delete;
  // Assignment operator (don't allow).
  StreamingMemoryOpInserter &
  operator=(const StreamingMemoryOpInserter &rhs) = delete;

  // Constructor.
  StreamingMemoryOpInserter(Graph &graph,
                            int64_t replicationFactor,
                            int num_stages,
                            int num_phases);

  // Add streaming memory operations.
  void apply();

  // Set up data structures to look up the schedule of ops and tensors
  void createTensorSchedule();

  // Find the related weight tensor downstream of the tensorSet
  Tensor *findRelatedVarTensor(std::vector<Tensor *> front);

  // Correct attributes on replicated reduction operations connected to the
  // optimizer
  void updateReplicatedOperations();

  // Correct attributes on optimizer operations
  void updateOptimizerOperations();

  // Determine where to place a tensor (used by StreamingMemory).
  TensorLocation determineTensorLocation(Tensor *tensor) const;
  // Sanity checking helper function (used by StreamingMemory).
  void sanitizePlacementAnnotation(Op *op, ExecutionPhase phase) const;

  // Deduct and set the priority on the op, based on
  // op type, phased execution and the schedule for the
  // phased execution
  static void setPriority(Op *op,
                          bool isPhased,
                          bool onDemandOptimizerState,
                          ExecutionPhaseSchedule schedule);

protected:
  // Types used.
  using Ops               = std::vector<Op *>;
  using OpsSet            = std::set<Op *, POpCmp>;
  using TensorIds         = std::vector<TensorId>;
  using LoadedTensorMap   = std::map<TensorId, int>;
  using GatheredTensorMap = std::map<TensorId, int>;
  using OpScheduleMap     = std::map<Op *, int64_t, POpCmp>;
  using TensorSchedule    = std::vector<Tensor *>;
  using TensorSet         = std::set<Tensor *, PTensorCmp>;

  class TensorStreamingContext {
  public:
    TensorStreamingContext();

    TensorStreamingContext(ExecutionContext,
                           OptionalExecutionPhase,
                           ScheduledPreLoss);

    bool operator<(const TensorStreamingContext &rhs) const;
    bool operator==(const TensorStreamingContext &rhs) const;
    bool operator!=(const TensorStreamingContext &rhs) const;

    // Annotates if a tensor is live, loaded, stored or gathered in:
    // 1. The execution context (always active)
    ExecutionContext context;
    // 2. The execution phase
    //    (for phased execution in the ExecutionContext::Normal only)
    OptionalExecutionPhase phase;
    // 3. The pre/post loss schedule
    //    (for non-phased execution in the ExecutionContext::Normal only)
    ScheduledPreLoss preLoss = ScheduledPreLoss::Undefined;
  };

  friend std::ostream &operator<<(std::ostream &output,
                                  const TensorStreamingContext &c);

  // The information required to apply changes necessary for this operation
  // related to the input tensor.
  class ConsumerOpConfig {
  public:
    ConsumerOpConfig(Tensor *tensor_, Op *op_, std::vector<InIndex> inIndices_)
        : tensor(tensor_), op{op_}, inIndices(inIndices_) {}

    // The originally consumed tensor
    Tensor *tensor;

    // The op to reconfigure
    Op *op;

    // The input indices of this tensor consumer
    std::vector<InIndex> inIndices;

    bool operator==(const ConsumerOpConfig &rhs) const;
  };

  using ConsumerOpConfigs = std::vector<ConsumerOpConfig>;

  class TensorStreamingConfig {
  public:
    // Flag to denote whether this is a producer context
    bool producer = false;
    // Flag to denote whether the tensor is live in this context
    bool live = false;
    // Virtual graph ID for this tensor.
    OptionalVGraphId streamingVGID;
    // Ops that are modifiers
    Ops modifiers;
    // Ops that are consumers
    ConsumerOpConfigs consumers;
    // Flags that denote whether the tensor needs to be loaded/stored/gathered
    bool load   = false;
    bool store  = false;
    bool gather = false;
  };

  // TensorStreamingContext: Where to apply graph streaming
  // TensorStreamingConfig: How to do apply graph streaming
  using TensorStreamingMap =
      std::map<TensorStreamingContext, TensorStreamingConfig>;

  // The information required to apply changes necessary for this tensor.
  class TensorConfig {
  public:
    // Constructor.
    TensorConfig(Graph &graph)
        : tensor{}, rootVarTensor{}, producerOp{}, consumerOps{}, location{},
          streamingMap{}, settings{graph, ""},
          ioSchedule(ExecutionPhaseIOSchedule::Preload),
          schedule(ExecutionPhaseSchedule::Interleaving) {}

    // The tensor affected.
    Tensor *tensor;
    // The variable root tensor which this tensor is aliased from, if available
    Tensor *rootVarTensor;
    // The descendant tensors if this is a variable root tensor
    TensorSet descendantTensors;
    // The tensor's producer op, if any.
    Op *producerOp;
    // The tensor's consuming ops.
    ConsumerOpConfigs consumerOps;
    // The desired tensor location.
    TensorLocation location;
    // Where and how to apply graph streaming.
    TensorStreamingMap streamingMap;
    // The settings to use for new ops.
    Op::Settings settings;
    // The input/output schedule for the tensor
    ExecutionPhaseIOSchedule ioSchedule;
    // The optimizer schedule for the tensor (for weights)
    ExecutionPhaseSchedule schedule;
  };

  using TensorConfigMap = std::map<Tensor *, TensorConfig, PTensorCmp>;

  class ReplicationShardedTensors {
  public:
    void insert(TensorId shardId,
                TensorId gatheredId,
                TensorId tensorId,
                TensorId refId);

    std::set<TensorId> getShardTensorIds() const;
    std::set<TensorId> getGatheredTensorIds() const;

    bool hasShard(TensorId tensorId) const {
      return shardToTensors.find(tensorId) != shardToTensors.end();
    }

    bool hasGathered(TensorId tensorId) const {
      return gatheredToTensors.find(tensorId) != shardToTensors.end();
    }

    // Returns the sharded tensor id
    TensorId getShard(TensorId tensorId) const;

    // Returns the gathered tensor id
    TensorId getGathered(TensorId tensorId) const;

    // Returns the original tensor id
    // (tensor name before offloading was applied)
    TensorId getTensor(TensorId tensorId) const;

    // Returns the reference tensor id
    // (tensor that determines the replica sharded tensor layout)
    TensorId getReference(TensorId tensorId) const;

  private:
    // The tensor shard -> tensor map
    std::map<TensorId, std::tuple<TensorId, TensorId, TensorId>> shardToTensors;
    // The gathered tensor -> tensor map
    std::map<TensorId, std::tuple<TensorId, TensorId, TensorId>>
        gatheredToTensors;
  };

  // Add streaming memory operations pertaining to one tensor.
  void applyTensor(Tensor *tensor, ReplicationShardedTensors &rtsTensors);

  // Reconfigure optimizer for replicated tensor sharding
  void applyReplicatedOptimizerSharding(ReplicationShardedTensors &rtsTensors);

  // Helper functions to populate tensor config (the functions
  // below and TensorConfig could possibly be put in their own class.)
  void getTensorOpSchedule(Tensor *tensor, TensorConfig &tensorConfig) const;
  void getTensorConfig(Tensor *tensor);
  void getRootVarTensor(Tensor *tensor, Tensor *&rootTensor) const;
  void getConsumerOps(Tensor *tensor, ConsumerOpConfigs &consumerOps) const;
  void filterAndSortConsumerOps(ConsumerOpConfigs &consumerOps) const;
  void getTensorLocation(Tensor *tensor, TensorLocation &location) const;
  void getTensorProducerStreamingConfig(Tensor *tensor,
                                        const TensorLocation &location,
                                        const Op *producerOp,
                                        TensorStreamingMap &streamingMap) const;
  void getTensorOptionalVGraphId(Tensor *tensor,
                                 const Op *producerOp,
                                 const ConsumerOpConfigs &consumerOps,
                                 OptionalVGraphId &streamingVGID) const;

  void getTensorStreamingConfig(Tensor *tensor);

  void getTensorSettings(Tensor *tensor,
                         const Op *producerOp,
                         const ConsumerOpConfigs &consumerOps,
                         Op::Settings &settings) const;
  void getModifiersInContext(Tensor *t,
                             const TensorStreamingContext context,
                             Ops &modifyingConsumerOps) const;
  void getAliasedModifiersInContext(Tensor *t,
                                    const TensorStreamingContext context,
                                    Ops &modifyingConsumerOps) const;

  // Helper to configure phase & priority on tensor loading operations
  // (Init -> RemoteLoad -> AllGather)
  void setLoadingOpPhaseAndPriority(Op *op,
                                    const Tensor *const tensor,
                                    const TensorConfig &tensorConfig,
                                    const TensorStreamingContext &context);

  // Helper functions to insert a RemoteLoadOp.
  RemoteLoadOp *insertRemoteLoadOp(const TensorConfig &tensorConfig,
                                   const TensorStreamingContext context,
                                   TensorId &loadedTensorId);

  // Helper functions to insert a RemoteStoreOp.
  RemoteStoreOp *insertRemoteStoreOp(const TensorConfig &tensorConfig,
                                     const TensorStreamingContext context,
                                     const TensorId &loadedTensorId);

  // Helper function to insert a ReplicatedAllGatherOp.
  ReplicatedAllGatherOp *
  insertReplicatedAllGatherOp(const TensorConfig &tensorConfig,
                              const TensorStreamingContext context,
                              const TensorId &loadedTensorId,
                              TensorId &gatheredTensorId,
                              const TensorId &referenceTensorId);

  // Helper function to insert a ReplicatedReduceScatter.
  ReplicatedReduceScatterOp *
  insertReplicatedReduceScatterOp(const TensorConfig &tensorConfig,
                                  const TensorStreamingContext context,
                                  const TensorId &inTensorId,
                                  const TensorId &outTensorId,
                                  const TensorId &weightTensorId);

  // Execution mode helper functions.
  // Phased execution: Ops are annotated with the ExecutionPhase attribute that
  // control device placement and the streaming memory schedule
  bool isPhasedExecution() const;

  // Sanity checking functions.
  void sanitizeOps() const;
  void verifyPlacementConsistency(const Op *op) const;

  // Log phase config for a tensor.
  void logTensorStreamingConfig(const TensorConfig &tensorConfig) const;

  // Helper function a remote buffer pointer tensor belonging to tensorId, if it
  // does not exist yet, and add it to remoteArgIds
  TensorId getRemoteArg(TensorId tensorId);

  // Helper functions for generating tensor IDs.
  TensorId generateInitTensorId(TensorId id);
  TensorId generateLoadedTensorId(TensorId id);
  TensorId getPreviousLoadedTensorId(TensorId id);
  TensorId generateGatheredTensorId(TensorId id);

  // Static helper functions.
  static bool
  tooSmallForOffChip(const TensorLocationSettings &tensorLocationSettings,
                     Tensor *tensor);
  static bool tooSmallForReplicatedTensorSharding(
      const TensorLocationSettings &tensorLocationSettings,
      Tensor *tensor);

  // The graph the transform operates on.
  Graph &graph;

  // The Op schedule before this transform inserts new ops and tensors
  OpScheduleMap opScheduleMap;

  // The tensor schedule (determines processing order)
  // before this transform inserts new ops and tensors
  TensorSchedule tensorSchedule;

  // Number of graph replications.
  const int64_t replicationFactor;

  // Number of sets of IPUs.
  const int num_stages;

  // Number of execution phases.
  const int num_phases;

  // A list of remote buffer pointer tensor ids.
  TensorIds remoteArgIds;

  // A map of tensor configurations
  TensorConfigMap tensorConfigs;

  // A map of loaded tensor ID counters
  LoadedTensorMap loadedTensorCounter;

  // A map of gathered tensor ID counters
  GatheredTensorMap gatheredTensorCounter;
};

} // namespace popart

#endif
