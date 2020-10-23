#ifndef GUARD_NEURALNET_BATCHSERIALIZE_HPP
#define GUARD_NEURALNET_BATCHSERIALIZE_HPP

#include <queue> // we use a priority_queue
#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

// Batch serialization:
// Serializes Ops in the forward pass along the batch dimension,
// starting from data inputs. The data/activation path is sliced along the
// batch dimension, and concatenated again for ops that do not support batch
// serialization, such as Losses and BatchNorm.
// Crossing boundaries such as ExecutionPhases, VirtualGraphs and Pipeline
// stages also causes the batch to be concatenated again.
// The backward pass is grown as normal, but the second batch serialization pass
// will look for local graph isomorphisms to ensure each batch serialized
// sequence (for each batch) is scheduled identically when possible,
// which improves the outlining outcome.
//
// Before transformation:
//           w0                          w1
//           |                           |
//   data - MatMul - ReLU - BatchNorm - MatMul - Loss
//
// After transformation (batch serialization factor 4):
//
//        data [batch(4), c, h, w]
//          |
//          +-----------------+-----------------+-----------------+
//          |                 |                 |                 |
//       DynamicSlice(0)   DynamicSlice(1)   DynamicSlice(2)   DynamicSlice(3)
//          |                 |                 |                 |
//    w0 - MatMul       w0 - MatMul       w0 - MatMul       w0 - MatMul
//          |                 |                 |                 |
//         ReLU              ReLU              ReLU              ReLU
//          |                 |                 |                 |
// Init- DynamicUpdate(0)- DynamicUpdate(1)- DynamicUpdate(2)- DynamicUpdate(3)
//                                                                |
//                                                              BatchNorm
//                                                                |
//          +-----------------+-----------------+-----------------+
//          |                 |                 |                 |
//       DynamicSlice(0)   DynamicSlice(1)   DynamicSlice(2)   DynamicSlice(3)
//          |                 |                 |                 |
//    w1 - MatMul       w1 - MatMul       w1 - MatMul       w1 - MatMul
//          |                 |                 |                 |
// Init- DynamicUpdate(0)- DynamicUpdate(1)- DynamicUpdate(2)- DynamicUpdate(3)
//                                                                |
//                                                               Loss

namespace popart {

using IpuNumber = int64_t;

IpuNumber getIpuNumber(const Op *op);

class BatchSerialize : public Transform {
public:
  static std::size_t id(int);

  BatchSerialize(int pass_) : Transform(), pass(pass_) {}
  virtual ~BatchSerialize() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(pass); }

  virtual std::string getName() const override final {
    return "BatchSerialize";
  }

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

  using TensorContext = std::tuple<VGraphId, ExecutionPhase, PipelineStage>;
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
  int64_t getLocalIsoScore(
      std::map<std::tuple<Op *, Op *, int>, int64_t> &cachedIsoScores,
      const OpToSection &opToSection,
      const OpToPosition &opToPosition,
      std::map<Op *, SubgraphEquivId> &opSubgraphEquivId,
      std::pair<Op *, Op *> ops,
      std::set<std::pair<Op *, Op *>> &visitedOps,
      int maxDepth,
      bool cached) const;

  OpId reshapeForSlice(Graph &graph,
                       Op::Settings settings,
                       TensorId inId,
                       Shape newShape,
                       TensorId newId,
                       OptionalBatchSerializedPhase bsp) const;

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
  void addParallelizationConstraints(Graph &graph,
                                     const PositionsToOp &positionToOp) const;

  // Get the last RemoteLoadOp that copies from IO to compute tiles for a given
  // section and phase, if one exists, and nullptr otherwise.
  Op *getLastRemoteLoad(const PositionsToOp &positionsToOp,
                        const Section section,
                        const BatchSerializedPhase phase) const;

  // Get the last IoTileCopyOp that copies from IO to compute tiles for a given
  // section and phase, if one exists, and nullptr otherwise.
  Op *getLastIoTileCopyToCompute(const PositionsToOp &positionsToOp,
                                 const Section section,
                                 const BatchSerializedPhase phase) const;
  // Get the first IoTileCopyOp that copies from compute to IO tiles for a given
  // section and phase, if one exists, and nullptr otherwise.
  Op *getFirstIoTileCopyToIo(const PositionsToOp &positionsToOp,
                             const Section section,
                             const BatchSerializedPhase phase) const;

  // Get the first compute operation for a given section and phase, if one
  // exists, and nullptr otherwise.
  Op *getFirstComputeOp(const PositionsToOp &positionsToOp,
                        const Section section,
                        const BatchSerializedPhase phase) const;

  // Try and reorder map to be more ameniable to overlapping compute & IO.
  void tryToMakeAmenableToParallelization(Graph &graph,
                                          PositionsToOp &positionsToOp) const;

  int pass;
};

} // namespace popart

#endif
