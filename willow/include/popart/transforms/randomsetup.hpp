
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RANDOMSETUP_HPP
#define GUARD_NEURALNET_RANDOMSETUP_HPP

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include <popart/pointercomparators.hpp>
#include <popart/transforms/transform.hpp>
#include <popart/vendored/optional.hpp> // IWYU pragma: keep

#include "popart/graphid.hpp"
#include "popart/names.hpp"

namespace popart {
// There are a number of operations that exhibit random behaviour in PopART.
// These 'random ops' are all derived from RandomBaseOp and require a seed
// tensor as input ([2], UINT32). This seed tensor is passed to the random
// number generator to mimic random behaviour. The random seed input of a random
// op may be set by the user, in which case there is nothing we need to do here.
// This transformation adds seed tensors for those random ops that do not yet
// have one.
//
// There are a number of requirements this transformation needs to meet:
//
// * To ensure distinct random behaviour, the seeds we assign to random ops
//   in the forward pass should be such that no two forward-pass executions
//   share a random seed. This includes cases where the random op lives
//   in a subgraph and that subgraph is called twice --the calls should not
//   share a random seed-- and cases where it is ran multiple times in the
//   body of a loop -- the iterations should not share seeds.
// * Random behaviour for each input batch should be distinct from previous
//   batches. Seeds should not be re-used in a deterministic way across batches
// * Random ops that are recomputed during the backwards pass as well as
//   the gradient ops belonging to random ops, should exhibit the same random
//   behaviour as the associated forward random op.
// * Random ops should be outlinable.
// * Solution must be compatible with batch serialisation.
// * Solution must not unduly fix the schedule.
// * Solution must work with random ops in subgraph.
// * Solution must work with implicit recomputation. As only main-graph ops
//   are checkpointed this implies we need the pass seeds to subgraphs.
//
//   NOTE: This transform is ran after the forward pass. Recomputation and
//   growing of the backwards pass is done later in the pipeline.
//
// To achieve the requirements above, this transform introduces a number of new
// ops the IR, which we will now describe.
//
//
// GetRandomSeedOp (GRSOp)
// =======================
//
// This op lives in the main graph and produces a start seed from which all
// other seeds (for a given batch) are derived in a deterministic way. The
// produced seed is first a tensor streamed from the host but it's value is
// incremented for every batch (to ensure distinct batches have distinct random
// behaviour).
//
//  [randomSeed___fromHost] = [2 UINT32] Initial seed streamed from host.
//   |
//   v
// +------------------+
// | GetRandomSeedOp  |
// +------------------+
//   |
//   v
//  [randomSeed___updated]  = [2 UINT32] Base seed for a particular batch.
//
//
// ModifySeedOp (MSOp)
// ===================
//
// Each op that requires a random seed uses a ModifySeedOp to produce the
// seed. This op takes a base_seed (either the seed produced by GetRandomSeedOp
// or the seed produced by a ModifySeedOp associated with the subgraph op that
// called the subgraph the random op is in) and a constant that is unique within
// the subgraph and subsequently uses a random choice to determine a seed for
// the op to use. Note that for ops in the body of loops the base_seed needs to
// be loop-carried and incremented to avoid use of repeated seeds in loop
// iterations.
//
//  [base_seed]             = [2 UINT32] Input seed.
//   | [in_gen]             = [1 UINT32] Constant that is unique within a
//   |  |                                subgraph.
//   v  v
// +------------------+
// | ModifySeedOp     |
// +------------------+
//   |
//   v
//  [op_seed]               = [2 UINT32] Seed to be used by a random op.
//
// What GenerateSeedOp does is generate a seed output as follows:
//
//   [op_seed] = [base_seed[0], randint(base_seed[1] + in_gen)]
//
// NOTE: We determine the seed via a random choice rather than a simple
// increment because without this it is difficult to ensure that the random
// seeds used in different subgraphs are not deterministically the same.
//
// To explain what this transform does, consider an IR that look like follows
// just after the forward pass is done:
//
//
//     [main_in0]                                 [A_in0]
//       |  [main_in1]                             |
//       |   |                                     |
// +-----|---|--<main>-------------------+   +-<A>-|--------------------------+
// |     |   |                           |   |     |                          |
// |     v    \____                      |   |     |                          |
// |    Op1        \                     |   |     |                          |
// |     |          |                    |   |     V                          |
// |     v          |                    |   |    RandomOp3                   |
// |    RandomOp1   |                    |   |     |                          |
// |     |          |                    |   |     |                          |
// |     |          |                    |   |     |                          |
// |     v          |                    |   +-----|--------------------------+
// |    LoopOp1<A>  |                    |         v
// |     |   _______/                    |        [A_out0]
// |     |  /                            |
// |     v v                             |
// |    Op2                              |
// |     |                               |
// |     v                               |
// |    CallOp1<B>                       |
// |     |                               |
// |     |                               |        [B_in0]
// |     |                               |         |
// |     v                               |   +-<B>-|--------------------------+
// |    CallOp2<B>                       |   |     |                          |
// |     |                               |   |     v                          |
// |     |    [custom_seed]              |   |    Op3                         |
// |     |     |                         |   |     |                          |
// |     v     v                         |   |     |                          |
// |    RandomOp2                        |   |     v                          |
// |     |                               |   |    RandomOp4                   |
// |     |                               |   |     |                          |
// +-----|-------------------------------+   +-----|--------------------------+
//       v                                         v
//      [main_out0]                               [C_out0]
//
//
// In the first RandomSetup pass, the transform does the following:
//
// * Add a GetRandomSeedOp to the main graph.
// * For each subgraph, work out which ops need a random seed and add a
//   constant that is unique for the subgraph and a ModifyRandomSeeOp to the
//   subgraph and connect it up.
//
//
//     [main_in0]                                 [A_in0]
//       |  [main_in1]                             |              [base_seed]
//       |   |                                     |               |
// +-----|---|--<main>-------------------+   +-<A>-|---------------|----------+
// |     |   |    [randomSeed___fromHost]|   |     |               |          |
// |     v    \____                 |    |   |     |          0    |<------.  |
// |    Op1        \    0         GRSOp  |   |     |          |    |       |  |
// |     |          |   |           |    |   |     V          v    |  1    |  |
// |     v          |   v  [...__updated]|   |    RandomOp3<-MRSOp-|  |    |  |
// |    RandomOp1<--)--MRSOp<------'|    |   |     |               |  v    |  |
// |     |          |   1           |    |   |     |              MRSOp    |  |
// |     |          |   |           |    |   |     |               |-------'  |
// |     v          |   v           |    |   +-----|---------------|----------+
// |    LoopOp1<A><-)--MRSOp<------'|    |         v               v
// |     |   _______/               |    |        [A_out0]       [base_seed_out]
// |     |  /                       |    |                        (loop carried)
// |     v v                        |    |
// |    Op2             2           |    |
// |     |              |           |    |
// |     v              v           |    |
// |    CallOp1<B><----MRSOp<------'|    |
// |     |                          |    |
// |     |              3           |    |        [B_in0]          [base_seed]
// |     |              |           |    |         |                |
// |     v              v           |    |   +-<B>-|----------------|---------+
// |    CallOp2<B><----MRSOp<------'     |   |     |                |         |
// |     |                               |   |     v                |         |
// |     |    [custom_seed]              |   |    Op3               |         |
// |     |     |                         |   |     |         0      |         |
// |     v     v                         |   |     |         |      |         |
// |    RandomOp2                        |   |     v         v      |         |
// |     |                               |   |    RandomOp4<-MRSOp-'          |
// |     |                               |   |     |                          |
// +-----|-------------------------------+   +-----|--------------------------+
//       v                                         v
//      [main_out0]                               [C_out0]
//
// NOTE: We separate the operations described above (except GetRandomSeedOp)
// into orthogonal ops for every distinct combination of virtual graph ID and
// pipeline stage. We call each such logic tree a 'strand'. This is a design
// decision to try and avoid unperformant copies of random seeds between IPUs at
// unfortunate times.
//
// NOTE: We don't use Poplar's built-in seedModifier parameter for this
// because this results in random operations that are not outlinable.
//

// Forward declarations.
class RandomBaseOp;
class SubgraphOp;
class Graph;
class Ir;
class Op;
class TensorInfo;

class RandomSetup : public Transform {
public:
  static std::size_t id();

  RandomSetup() : Transform() {}
  virtual ~RandomSetup() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "RandomSetup"; }

  static bool hasRandomSeed(const Ir &ir);
  static bool requiresRandomSeed(const Ir &ir);
  static TensorId getStreamedSeedTensorId();

protected:
  // Type representing the 'IPU' an op lives on.
  using GraphIds                = std::vector<GraphId>;
  using Strand                  = std::tuple<VGraphId, PipelineStage>;
  using Strands                 = std::vector<Strand>;
  using StrandToTensorId        = std::map<Strand, TensorId>;
  using OpToStrands             = std::map<Op *, std::vector<Strand>, POpCmp>;
  using OpToStrandToTensorId    = std::map<Op *, StrandToTensorId, POpCmp>;
  using GraphToInIndex          = std::map<GraphId, InIndex>;
  using GraphToOutIndex         = std::map<GraphId, OutIndex>;
  using GraphToStrands          = std::map<GraphId, Strands>;
  using GraphToOpToStrands      = std::map<GraphId, OpToStrands>;
  using GraphToStrandToTensorId = std::map<GraphId, StrandToTensorId>;
  using StrandsMapAndRandomOpsMap =
      std::tuple<GraphToStrands, GraphToOpToStrands>;
  using InAndOutBaseSeedMap =
      std::tuple<GraphToStrandToTensorId, GraphToStrandToTensorId>;
  using FirstSeedInIndexMapAndFirstSeedOutIndexMap =
      std::tuple<GraphToInIndex, GraphToOutIndex>;

  // A struct that holds information as to how to apply the transform.
  struct Config {
    bool setVirtualGraphIds;
    bool setExecutionPhases;
    bool setPipelineStages;

    // The strands required for each graph.
    GraphToStrands strandsMap;
    // The ops that require a random seed for each graph/strand. This list
    // contains both ops derived from RandomBaseOp as well as ops that call a
    // subgraph that has random behaviour.
    GraphToOpToStrands randomOpsMap;
    // The TensorId each graph/strand combination should use as the base seed
    // input to their respective ModifyRandomSeedOps.
    GraphToStrandToTensorId inBaseSeedMap;
    // The TensorId each graph/strand combination should use as a base seed
    // output (used to loop-carry the base seed).
    GraphToStrandToTensorId outBaseSeedMap;
    // The order in which graphs should be subject to application of changes.
    GraphIds graphApplyOrder;
    // Desired location of base in seeds in graph inputs.
    GraphToInIndex firstSeedInIndexMap;
    // Desired location of base out seeds in graph outputs.
    GraphToOutIndex firstSeedOutIndexMap;
  };

  // Top-level helper function to determine transformation parameters.
  Config determineConfig(const Ir &ir) const;
  // Sub-helper function to determine whether to set virtual graph ids.
  bool determineSetVirtualGraphIds(const Ir &ir) const;
  // Sub-helper function to determine whether to set execution phases.
  bool determineSetExecutionPhases(const Ir &ir) const;
  // Sub-helper function to determine whether to set pipeline stages ids
  bool determineSetPipelineStages(const Ir &ir) const;
  // Sub-helper function to identity op strands and random ops in graphs.
  StrandsMapAndRandomOpsMap
  determineStrandsMapAndRandomOpsMap(const Ir &ir) const;
  // Sub-helper function to get all ops derived from RandomBaseOp for a graph.
  OpToStrands getInitStrandToOps(const Graph &graph) const;
  // Sub-helper function to determine names of seeds for every graph/strand.
  InAndOutBaseSeedMap
  determineBaseSeedsMaps(const Ir &ir, const GraphToStrands &strandsMap) const;
  // Sub-helper function to determine InIndices/OutIndices.
  FirstSeedInIndexMapAndFirstSeedOutIndexMap
  determineSeedIndexMaps(const Ir &ir) const;
  // Sub-helper function to determine constraints governing the application
  // order.
  GraphIds determineGraphApplyOrder(const Ir &ir) const;

  // Top-level helper function to log transformation parameters.
  void logConfig(const Ir &ir, const Config &cfg) const;

  // Top-level helper function to transform a graph.
  void applyToGraph(Graph &graph, const Config &cfg) const;
  // Sub-helper to make sure base seeds are available.
  void addBaseSeeds(Graph &graph, const Config &cfg) const;
  // Helper function to add GetRandomSeed op.
  void addGetRandomSeedOp(Ir &ir, const Config &cfg) const;
  // Sub-helper to add one ModifyRandomSeedOp.
  TensorId addModifyRandomSeedOp(Graph &graph,
                                 const Config &cfg,
                                 const Strand &strand,
                                 uint32_t modifier,
                                 nonstd::optional<TensorId> opSeedId,
                                 const std::string &seedReasonStr) const;
  // Sub-helper to add ModifyRandomSeedOps.
  OpToStrandToTensorId addModifyRandomSeedOps(Graph &graph,
                                              const Config &cfg) const;
  // Sub-helper to modify CallOps, LoopOps, etc.
  void connectOp(Graph &graph,
                 const Config &cfg,
                 const StrandToTensorId &opSeeds,
                 Op *op) const;
  // Sub-helper to modify RandomBaseOps.
  void connectRandomBaseOp(Graph &graph,
                           const Config &cfg,
                           const StrandToTensorId &opSeeds,
                           RandomBaseOp *op) const;
  // Sub-helper to modify SubgraphOps. Here, inputOffset/outputOffset is the
  // difference between the subgraph's in/out index and the subgraph op's in/out
  // index. A value of N indicates "subgraph index = subgraph op index + N".
  void connectSubgraphOp(Graph &graph,
                         const Config &cfg,
                         const StrandToTensorId &opSeeds,
                         SubgraphOp *op,
                         int inputOffset,
                         int outputOffset) const;

  // Helper function: are there any random ops in the IR?
  static bool hasRandomOps(const Ir &ir);
  // Helper function to determine the 'placement' for an op.
  static Strand getStrand(const Op *op);

  // Add strand info to tensor id.
  static TensorId getTensorIdForStrand(const TensorId &id,
                                       const Strand &strand);

  /*
  // Helper function to determine the op strands in a graph.
  static Strands getStrands(const Graph &graph, const Config &cfg);
  */

  static TensorInfo seedTensorInfo;

  // Helper function to serialise an Strand value.
  friend std::ostream &operator<<(std::ostream &out,
                                  const RandomSetup::Strand &strand);
};

// Helper function to serialise an Strand value.
std::ostream &operator<<(std::ostream &out, const RandomSetup::Strand &strand);

} // namespace popart

#endif
