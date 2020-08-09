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

  // Determine where to place a tensor (user by PingPong).
  TensorLocation determineTensorLocation(Tensor *tensor) const;
  // Sanity checking helper function (used by PingPong).
  void sanitizePlacementAnnotation(Op *op, PingPongPhase phase) const;

private:
  // Types used.
  using Ops               = std::vector<Op *>;
  using SetupOps          = std::set<Op *, POpCmp>;
  using TensorIds         = std::vector<TensorId>;
  using TensorPhase       = std::pair<TensorId, int64_t>;
  using RemoteStoreOpData = std::pair<TensorId, RemoteStoreOp *>;
  using RemoteLoadOpData  = std::tuple<TensorId, TensorId, RemoteLoadOp *>;
  using TensorStoreMap    = std::map<TensorPhase, RemoteStoreOpData>;
  using TensorLoadMap     = std::map<TensorPhase, RemoteLoadOpData>;

  class TensorPhaseConfig {
  public:
    // Pingpong phase of of the op that produced the tensor, if available/set.
    // This field is used to derive other fields and is logged.
    OptionalPingPongPhase producerPingPongPhase;
    // Virtual graph ID for this tensor.
    OptionalVGraphId loadStoreVGID;
    // The pingpong phases in which this tensor is 'live'.
    std::set<PingPongPhase> livePhases;
    // A mapping from pingpong phases to a list of ops that modify this tensor
    // in the respective pingpong phase.
    std::map<PingPongPhase, Ops> modifiersInPhase;
    // A mapping from pingpong phases to lists of consumer ops that are computed
    // in the respective phases.
    std::map<PingPongPhase, Ops> consumersInPhase;
    // A mapping from the various pingpong phases to flags that denote whether
    // the tensor needs to be loaded/store in that phase of pingpong.
    std::map<PingPongPhase, std::pair<bool, bool>> loadStoreInPhase;
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
          ioSchedule(PingPongIOSchedule::Preload),
          optimizerSchedule(PingPongOptimizerSchedule::Interleaving) {}

    // The tensor affected.
    Tensor *tensor;
    // The tensor's producer op, if any.
    Op *producerOp;
    // The tensor's consuming ops.
    Ops consumerOps;
    // The desired tensor location.
    TensorLocation location;
    // The pingpong config of the tensor.
    TensorPhaseConfig phaseConfig;
    // The settings to use for new ops.
    Op::Settings settings;
    // The input/output schedule for the tensor
    PingPongIOSchedule ioSchedule;
    // The optimizer schedule for the tensor (for weights)
    PingPongOptimizerSchedule optimizerSchedule;
  };

  // Add streaming memory operations pertaining to one tensor.
  void applyTensor(Tensor *tensor, SetupOps &opsToSetup);

  // Helper functions to populate tensor config (the functions
  // below and TensorConfig could possibly be put in their own class.)
  void getTensorSchedule(Tensor *tensor, TensorConfig &tensorConfig) const;
  void getTensorConfig(Tensor *tensor, TensorConfig &tensorConfig) const;
  void getOrderedConsumerOps(Tensor *tensor, Ops &consumerOps) const;
  void getTensorLocation(Tensor *tensor, TensorLocation &location) const;
  void getTensorPhaseConfig(Tensor *tensor,
                            const Op *producerOp,
                            const Ops &consumerOps,
                            TensorPhaseConfig &phaseConfig) const;
  void getTensorSettings(Tensor *tensor,
                         const Op *producerOp,
                         const Ops &consumerOps,
                         Op::Settings &settings) const;
  void getModifiersInPhase(Tensor *t,
                           const PingPongPhase phase,
                           Ops &modifyingConsumerOps) const;
  void getAliasedModifiersInPhase(Tensor *t,
                                  const PingPongPhase phase,
                                  Ops &modifyingConsumerOps) const;

  // Helper functions to insert a RemoteLoadOp.
  std::tuple<RemoteLoadOp *, ReplicatedAllGatherOp *>
  insertRemoteLoadOp(const TensorConfig &tensorConfig,
                     const PingPongPhase currentPingPongPhase,
                     TensorId &loadedTensorId,
                     TensorId &gatheredTensorId);

  // Helper functions to insert a RemoteStoreOp.
  RemoteStoreOp *insertRemoteStoreOp(const TensorConfig &tensorConfig,
                                     const PingPongPhase currentPingPongPhase,
                                     const TensorId &loadedTensorId);

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
  static TensorId generateLoadedTensorId(Tensor *tensor, int64_t load_index);
  static TensorId generateGatheredTensorId(Tensor *tensor, int64_t load_index);

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
  // Number of pingpong phases.
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
