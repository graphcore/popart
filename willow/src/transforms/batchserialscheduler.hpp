// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BATCHSERIALSCHEDULER_HPP
#define GUARD_NEURALNET_BATCHSERIALSCHEDULER_HPP

#include <queue> // we use a priority_queue
#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

struct BatchSerialTensorContext {
public:
  BatchSerialTensorContext() {}
  BatchSerialTensorContext(OptionalVGraphId vgraphId_,
                           OptionalExecutionPhase phase_,
                           OptionalPipelineStage stage_)
      : vgraphId(vgraphId_), executionPhase(phase_), pipelineStage(stage_) {}
  OptionalVGraphId vgraphId;
  OptionalExecutionPhase executionPhase;
  OptionalPipelineStage pipelineStage;
  bool operator<(BatchSerialTensorContext const &rhs) const;
  bool operator==(BatchSerialTensorContext const &rhs) const;
  bool operator!=(BatchSerialTensorContext const &rhs) const;
};

// Return the context of a tensor in the graph
BatchSerialTensorContext getBatchSerialTensorContext(const Op *op);

class BatchSerialScheduler {
public:
  BatchSerialScheduler(Graph &graph_);

  void apply();

private:
  using SubgraphEquivId     = std::string;
  using Position            = int64_t;
  using Section             = int64_t;
  using IsoScoreCache       = std::map<std::tuple<Op *, Op *, int>, int64_t>;
  using OpsBehindSection    = std::map<Section, std::vector<Op *>>;
  using OpToSection         = std::map<Op *, Section>;
  using OpToPosition        = std::map<std::pair<Section, BatchSerializedPhase>,
                                std::map<Op *, Position>>;
  using PositionsToOp       = std::map<std::pair<Section, BatchSerializedPhase>,
                                 std::map<Position, Op *>>;
  using PositionsToOpVector = std::vector<std::pair<Position, Op *>>;

  enum class TraceDirection { Forward = 0, Backward };

  class TraceFront {
  public:
    TraceFront(std::vector<Tensor *> tensors_, TraceDirection direction_);

    int64_t score() const;

    // Trace fronts with fewer producers/consumers first
    // (smaller chance of matching wrong isomorphic ops)
    bool operator<(const TraceFront &rhs) const;

    std::vector<Tensor *> tensors;
    TraceDirection direction;
  };

  // Calculates how similar two operations are in the context of a graph
  int64_t getLocalIsoScore(std::pair<Op *, Op *> ops,
                           std::set<std::pair<Op *, Op *>> &visitedOps,
                           int maxDepth,
                           bool cached);

  // Return fronts of tensors that can be used to find isomorphic ops
  std::priority_queue<TraceFront>
  findParallelTraceFronts(std::vector<Op *> schedule,
                          int64_t batchSerFactor) const;

  // Return true if we allow tweaking schedules by switching ops. We should only
  // allow switching when semantics are preserved.
  bool areSwappable(Graph &graph, Op *first, Op *second) const;

  // For a given vector of pairs of positions and ops, move ops (as selected by
  // isPushOp) as far foward the vector as is legal as defined by the pairwise
  // areOpsSwappable function.
  void pushEarlier(PositionsToOpVector &vec,
                   std::function<bool(Op *)> isPushOp,
                   std::function<bool(Op *)> considerSwappingWith,
                   std::function<bool(Op *, Op *)> areSwappable) const;

  // For a given vector of pairs of positions and ops, move ops (as selected by
  // isPushOp) as far back the vector as is legal as defined by the pairwise
  // areOpsSwappable function.
  void pushLater(PositionsToOpVector &vec,
                 std::function<bool(Op *)> isPushOp,
                 std::function<bool(Op *)> considerSwappingWith,
                 std::function<bool(Op *, Op *)> areSwappable) const;

  // Add intra-batch parallelization constraints as topological constraints to
  // attempt to encourage parallelization between batches.
  void addParallelizationConstraints(Graph &graph) const;

  // Get the last RemoteLoadOp that copies from IO to compute tiles for a given
  // section and phase, if one exists, and nullptr otherwise.
  Op *getLastRemoteLoad(const Section section,
                        const BatchSerializedPhase phase) const;

  // Get the last IoTileCopyOp that copies from IO to compute tiles for a given
  // section and phase, if one exists, and nullptr otherwise.
  Op *getLastIoTileCopyToCompute(const Section section,
                                 const BatchSerializedPhase phase) const;
  // Get the first IoTileCopyOp that copies from compute to IO tiles for a given
  // section and phase, if one exists, and nullptr otherwise.
  Op *getFirstIoTileCopyToIo(const Section section,
                             const BatchSerializedPhase phase) const;

  // Get the first compute operation for a given section and phase, if one
  // exists, and nullptr otherwise.
  Op *getFirstComputeOp(const Section section,
                        const BatchSerializedPhase phase) const;

  // Try and reorder map to be more ameniable to overlapping compute & IO.
  void tryToMakeAmenableToParallelization();

  Graph &graph;
  std::set<Op *> equivProcessedOps;
  OpToSection opToSection;
  OpToPosition opToPosition;
  PositionsToOp positionToOp;
  OpsBehindSection opsBehindSection;
  IsoScoreCache cachedIsoScores;
  std::map<Op *, int64_t> opScheduleIndex;
  std::map<Op *, SubgraphEquivId> opSubgraphEquivId;
};

} // namespace popart

#endif
