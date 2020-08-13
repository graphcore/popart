// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STREAMING_MEMORY_OP_INSERTER_HPP
#define GUARD_NEURALNET_STREAMING_MEMORY_OP_INSERTER_HPP

#include <map>

#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class RemoteLoadOp;
class RemoteStoreOp;
class ReplicatedAllGatherOp;

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

  // Determine where to place a tensor (used by StreamingMemory).
  TensorLocation determineTensorLocation(Tensor *tensor) const;
  // Sanity checking helper function (used by StreamingMemory).
  void sanitizePlacementAnnotation(Op *op, ExecutionPhase phase) const;

private:
  // Types used.
  using Ops               = std::vector<Op *>;
  using SetupOps          = std::set<Op *, POpCmp>;
  using TensorIds         = std::vector<TensorId>;
  using TensorPhase       = std::pair<TensorId, int64_t>;
  using RemoteStoreOpData = std::pair<TensorId, RemoteStoreOp *>;
  using RemoteLoadOpData  = std::pair<TensorId, RemoteLoadOp *>;
  using TensorStoreMap    = std::map<TensorPhase, RemoteStoreOpData>;
  using TensorLoadMap     = std::map<TensorPhase, RemoteLoadOpData>;

  class TensorPhaseConfig {
  public:
    // Execution phase of of the op that produced the tensor, if available/set.
    // This field is used to derive other fields and is logged.
    OptionalExecutionPhase producerExecutionPhase;
    // Virtual graph ID for this tensor.
    OptionalVGraphId loadStoreVGID;
    // The execution phases in which this tensor is 'live'.
    std::set<ExecutionPhase> livePhases;
    // A mapping from execution phases to a list of ops that modify this tensor
    // in the respective execution phase.
    std::map<ExecutionPhase, Ops> modifiersInPhase;
    // A mapping from execution phases to lists of consumer ops that are
    // computed in the respective phases.
    std::map<ExecutionPhase, Ops> consumersInPhase;
    // A mapping from the various execution phases to flags that denote whether
    // the tensor needs to be loaded/store in that phase of execution.
    std::map<ExecutionPhase, bool> loadInPhase;
    std::map<ExecutionPhase, bool> storeInPhase;
    std::map<ExecutionPhase, bool> gatherInPhase;
    // True if we need to load/store the tensor outside of the main loop. This
    // is used for RWS + OnChip weights.
    bool loadStoreOutOfPhase;
  };

  // The information required to apply changes necessary for this tensor.
  class TensorConfig {
  public:
    // Constructor.
    TensorConfig(Graph &graph)
        : tensor{}, producerOp{}, consumerOps{}, location{},
          phaseConfig{}, settings{graph, ""},
          ioSchedule(ExecutionPhaseIOSchedule::Preload),
          optimizerSchedule(ExecutionPhaseOptimizerSchedule::Interleaving) {}

    // The tensor affected.
    Tensor *tensor;
    // The tensor's producer op, if any.
    Op *producerOp;
    // The tensor's consuming ops.
    Ops consumerOps;
    // The desired tensor location.
    TensorLocation location;
    // The execution phase config of the tensor.
    TensorPhaseConfig phaseConfig;
    // The settings to use for new ops.
    Op::Settings settings;
    // The input/output schedule for the tensor
    ExecutionPhaseIOSchedule ioSchedule;
    // The optimizer schedule for the tensor (for weights)
    ExecutionPhaseOptimizerSchedule optimizerSchedule;
  };

  // Add streaming memory operations pertaining to one tensor.
  void applyTensor(Tensor *tensor, SetupOps &opsToSetup);

  // Helper functions to populate tensor config (the functions
  // below and TensorConfig could possibly be put in their own class.)
  void getTensorSchedule(Tensor *tensor, TensorConfig &tensorConfig) const;
  void getTensorConfig(Tensor *tensor, TensorConfig &tensorConfig) const;
  void getOrderedConsumerOps(Tensor *tensor, Ops &consumerOps) const;
  void getTensorLocation(Tensor *tensor, TensorLocation &location) const;
  void
  getTensorProducerExecutionPhase(Tensor *tensor,
                                  const Op *producerOp,
                                  OptionalExecutionPhase &producerPhase) const;
  void getTensorOptionalVGraphId(Tensor *tensor,
                                 const Op *producerOp,
                                 const Ops &consumerOps,
                                 OptionalVGraphId &loadStoreVGID) const;
  void getTensorPhaseConfig(Tensor *tensor,
                            const Op *producerOp,
                            const Ops &consumerOps,
                            const TensorLocation &location,
                            TensorPhaseConfig &phaseConfig) const;
  void getTensorSettings(Tensor *tensor,
                         const Op *producerOp,
                         const Ops &consumerOps,
                         Op::Settings &settings) const;
  void getModifiersInPhase(Tensor *t,
                           const ExecutionPhase phase,
                           Ops &modifyingConsumerOps) const;
  void getAliasedModifiersInPhase(Tensor *t,
                                  const ExecutionPhase phase,
                                  Ops &modifyingConsumerOps) const;

  // Helper functions to insert a RemoteLoadOp.
  RemoteLoadOp *insertRemoteLoadOp(const TensorConfig &tensorConfig,
                                   const OptionalExecutionPhase phase,
                                   TensorId &loadedTensorId);

  // Helper functions to insert a RemoteStoreOp.
  RemoteStoreOp *insertRemoteStoreOp(const TensorConfig &tensorConfig,
                                     const OptionalExecutionPhase phase,
                                     const TensorId &loadedTensorId);

  // Helper function to insert a ReplicatedAllGatherOp.
  ReplicatedAllGatherOp *
  insertReplicatedAllGatherOp(const TensorConfig &tensorConfig,
                              const ExecutionPhase phase,
                              const TensorId &loadedTensorId,
                              TensorId &gatheredTensorId);

  // Sanity checking functions.
  void sanitizeOps() const;
  void verifyPlacementConsistency(const Op *op) const;

  // Log phase config for a tensor.
  void logTensorPhaseConfig(const TensorConfig &tensorConfig) const;

  // Helper function a remote buffer pointer tensor belonging to tensorId, if it
  // does not exist yet, and add it to remoteArgIds
  TensorId getRemoteArg(TensorId tensorId);
  nonstd::optional<std::string> getPreviousLoadedTensorId(TensorId &id);

  // Helper functions for generating tensor IDs.
  static TensorId generateInitTensorId(Tensor *tensor);
  static TensorId generateLoadedTensorId(Tensor *tensor,
                                         OptionalExecutionPhase phase);
  static TensorId generateGatheredTensorId(Tensor *tensor,
                                           ExecutionPhase phase);

  // Static helper functions.
  static bool isLoadRequired(const TensorPhaseConfig &phaseConfig);
  static bool
  tooSmallForOffChip(const TensorLocationSettings &tensorLocationSettings,
                     Tensor *tensor);
  static bool tooSmallForReplicatedTensorSharding(
      const TensorLocationSettings &tensorLocationSettings,
      Tensor *tensor);

  // The graph the transform operates on.
  Graph &graph;

  // Number of graph replications.
  const int64_t replicationFactor;
  // Number of sets of IPUs.
  const int num_stages;
  // Number of execution phases.
  const int num_phases;

  // A list of remote buffer pointer tensor ids.
  TensorIds remoteArgIds;
  // Result of RemoteStoreOps already applied.
  TensorStoreMap tensorStoreMap;
  // Result of RemoteLoadOps already applied.
  TensorLoadMap tensorLoadMap;
};

} // namespace popart

#endif
