// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PIPELINE_HPP
#define GUARD_NEURALNET_PIPELINE_HPP

#include <popart/graph.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/restore.hpp>
#include <popart/transforms/mainloops.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

/**
 * A helper class for constructing the pipeline on a per-cycle basis
 *
 * After the flush/ramp down is completed, accumulated gradients can be applied
 * to the weight update
 */
class PipelineInfo {
public:
  PipelineInfo() = default;

  /**
   * Construct PipelineInfo
   * \param batchesPerStep     Batches per step
   * \param gradAcclFactor     Gradient accumulation factor
   * \param numPipelineStages  The number of pipeline stages
   * \param doGradAccl         Gradients are accumulated rather than applied to
   *                           the weights directly.
   * \param withStage          Map of pipeline stages to which stage to execute
   *                           with in sequence
   */
  PipelineInfo(int64_t batchesPerStep,
               int64_t gradAcclFactor,
               int64_t maxPipelineStage,
               bool doGradAccl,
               std::map<PipelineStage, PipelineStage> withStage);

  int64_t numStages;

  bool doGradAccl;

  /**
   * Struct describing a closed interval of pipeline cycles contained
   * in a pipeline phase
   */
  struct PipelinePhase {
    // [start, end]
    PipelineCycle start, end;
  };

  /**
   * The cycle interval for the fill phase
   */
  PipelinePhase fillPhase;

  /**
   * The cycle interval for the main phase (looped between fill and flush)
   */
  PipelinePhase mainPhase;

  /**
   * The cycle interval for the flush phase
   */
  PipelinePhase flushPhase;

  /**
   * The map of stages to the stage it should be executed with in sequence
   */
  std::map<PipelineStage, PipelineStage> withStage;

  /**
   * Get the number of independent stages in an interval of stages.
   * \param start First pipeline stage to check (inclusive)
   * \param end   Last pipeline stage to check (exclusive)
   * \return The number of independently executed pipeline stages
   */
  int numIndependentStages(PipelineStage start, PipelineStage end);

  /**
   * Checks if a \a stage stage needs to be executed in a \a cycle
   * \param pCycle the cycle to check
   * \param pStage the stage to check
   * \return true  if a stage is to be executed in a cycle
   */
  bool doStage(PipelineCycle pCycle, PipelineStage pStage) const;

  /**
   * The stage this stage should be executed with in sequence
   * \return previous pStage with which to execute this stage
   */
  PipelineStage executeWithStage(PipelineStage pStage) const;

  /**
   * Number of cycles in the main phase interval
   * \return Number of cycles, start and end inclusive (closed interval)
   */
  PipelineCycle getMainCycles() const {
    return mainPhase.end - mainPhase.start + 1;
  }
};

class Pipeline : public Transform {
public:
  static std::size_t id();

  Pipeline() : Transform() {}
  virtual ~Pipeline() override {}

  /**
   * Checks if the pipelining settings are valid and applies either
   * implicit or explicit pipelining transforms to the graph
   * \param graph top-level IR graph (main graph)
   * \return true if the transformation has changed the graph
   */
  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "Pipeline"; }

  /**
   * Implicit pipelining and implicit recompute only!
   * Test if the (implicit) recompute logic requires an inplace restored
   * version of a forward ActGrad tensor (from the stash)
   * \param op    the Op to check if it is convertible to RestoreInplaceOp and
   *              is required for (implicit) recompute
   * \return True if the inplace restore is required
   */
  static bool inplaceRestoreRequiredForRecompute(Op *op);

  /**
   * Implicit pipelining and implicit recompute only!
   * Check if implicit recompute is in conflict with implicit pipelining
   * when restoring a forward ActGrad tensor inplace
   * \param  op the Op to check
   * \param  in input index of the Op
   * \param  out output index of the Op
   * \return true if there is an inplace overwritng conflict
   */
  static bool inplaceRecomputationConflict(Op *op, InIndex in, OutIndex out);

  /**
   * Implicit recompute only!
   * This annotation pass will try to set the Ops between
   * the topologically final Checkpoints and the loss
   * to NOT be recomputed. This avoid a program where
   * operations are run twice in a row with no benefit to
   * liveness.
   * \param graph top-level IR graph (main graph)
   */
  static void setFinalFwdStageRecomputation(Graph &graph);

  /**
   * Check which stages should be executed with which other stage
   * \param ir IR from which to read the pipeline stages
   * \return   Map of pipeline stages to which stage to execute
   *           with in sequence.
   */
  static std::map<PipelineStage, PipelineStage> withStages(const Ir &ir);

  /**
   * Calculate the required stash size.
   * \param ir              The current IR
   * \param stashStage      The stage in which the stash is updated
   * \param maxRestoreStage The last stage in which the stash is restored
   * \return                Required number of stash entries
   */
  int getStashSize(const Ir &ir,
                   PipelineStage stashStage,
                   PipelineStage maxRestoreStage) const;

private:
  /**
   * Adds a restore operation to the graph
   * \param  graph Graph on which to add the restore operation
   * \param  stashSize Number of entries in the stash
   * \return RestoreOp instance
   */
  RestoreOp *addNewRestoreOp(Graph &graph, int64_t stashSize) const;

  /**
   * Adds a restore inplace operation to the graph
   * \param  graph Graph on which to add the restore operation
   * \param  stashSize Number of entries in the stash
   * \return RestoreInplaceOp instance
   */
  RestoreInplaceOp *addNewRestoreInplaceOp(Graph &graph,
                                           int64_t stashSize) const;

  /**
   * Add all required stash and restore operations to the graph, which link
   * forward and recompute/backward stages together
   * \param  graph top-level IR graph (main graph)
   * \return true
   */
  bool addStashRestoreOps(Graph &graph) const;

  /**
   * Transforms a linear set of pipeline stages into (explicit) pipeline cycles,
   * by inserting CallOps for each pipeline stage and unrolling the innermost
   * (explicit) loop partially.
   * \param  graph the innermost loop graph containing the pipeline operations
   *               with gradient accumulation enabled, the loop subgraph is the
   *               accumulation loop, otherwise it is the batches-per-step loop
   * \return true
   */
  bool applyExplicit(Graph &graph) const;
};

/**
 * \class ExplicitPipeline
 * \brief Transforms the graph to express the pipeline explicitly.
 *
 * .. warning::
 *
 *    Will not work with a nested scope (for example in stepLoop and
 *    gradAccumulation)
 *
 * See docs/notes/transforms/pipelining.md for a detailed description
 * of pipelining
 */
class ExplicitPipelineHelper {
public:
  /**
   * Create the helper and register the graph to operate on
   * \param innerLoopSubgraph_ the innermost loop graph containing the pipeline
   *                           operations with gradient accumulation enabled,
   *                           the loop subgraph is the accumulation loop,
   *                           otherwise it is the batches-per-step loop
   */
  ExplicitPipelineHelper(Graph &innerLoopSubgraph_);

  /**
   * Create the explicit pipeline.
   */
  void createExplicitPipeline();

private:
  /**
   * The innermost loop graph containing the pipeline
   * operations with gradient accumulation enabled,
   * the loop subgraph is the accumulation loop,
   * otherwise it is the batches-per-step loop
   */
  Graph &innerLoopSubgraph;

  /**
   * Reference to IR
   */
  Ir &ir;

  /**
   * Helper for constructing the pipeline on a per-cycle basis
   */
  PipelineInfo pInfo;

  /**
   * Pointer to Op which is the primary call site of the \a innerLoopSubgraph
   */
  LoopOp *pipelineMainLoop;

  /**
   * Struct describing how a clone of the graph containing the pipeline stages
   * maps to the original and vice versa.
   * The cloned graph is used as a reference while transforming the original
   * graph into fill, main and flush phases.
   */
  struct InnerLoopSubgraphClone {
    std::string clonedGraphId = "pipelineClone";
    std::map<OpId, OpId> originalGraphOpIdAndClonedGraphOpId;
    std::map<OpId, OpId> clonedGraphOpIdAndOriginalGraphOpId;
  } cloneSrct;

  /**
   * Struct containing maps to categorize any pipeline annotated Op
   * \a hostLoadOps:  instances of \a HostLoadOp
   * \a hostStoreOps: instances of \a HostStoreOp
   * \a ipuCopyOps:   instances of \a IpuCopyOp (cross-pipeline stage copies)
   * \a mainOps:      remaining operations
   */
  struct InnerLoopOpsCategories {
    std::map<PipelineStage, std::vector<OpId>> mainOps, hostLoadOps,
        hostStoreOps, ipuCopyOps;
  } innerLoopOpsCategories;

  /**
   * Struct to help link \a OpIds to stages and stages to \a IpuCopyOps
   */
  struct PipelineStageOpIdMaps {
    std::map<OpId, PipelineStage> opIdAndPipelineStage;
    std::map<PipelineStage, OpId> ipuCopyCallOps;
  } pipelineStageOpIdMaps;

  /**
   * Throw error if it is not possible to perform explicit pipelining.
   * Requirements:
   * - HostLoad/HostStore operations must be enabled
   * - Explicit main loops must be enabled
   * - Explicit recomputation must be enabled
   * - No anchors can be of type EveryN
   */
  void compatibilityChecks() const;

  /**
   * Fill the \a innerLoopOpsCategories
   */
  void categorizeInnerLoopOps();

  /**
   * Replace opsCategories with callOps
   *
   * \param  opsCategory  The ops to move into subgraphs and replace with
   * CallOps \param  subgraphPostfix Postfix to be used in the subgraph Id
   * \return Map of pipepline stage to replacement CallOps
   */
  std::map<PipelineStage, OpId> replaceOpsCategoriesWithCallOps(
      const std::map<PipelineStage, std::vector<OpId>> &opsCategory,
      std::string subgraphPostfix);

  /**
   * Create callOps of the innerLoopOpsCategories and fill pStageOpIdMaps.
   */
  void createCallOps();

  /**
   * Unroll the loop operator and creates the fill phase.
   *
   * \return A map between the TensorId of the innerLoopSubgraph and the
   *         TensorId belonging to the cloned Ops in the fill stage.
   *         These tensors represent the input to the main pipeline phase.
   */
  std::map<TensorId, TensorId> createFillPhase();

  /**
   * Modify the pipeline loop to adjust the main phase.
   *
   * \param lastOriginalAndClonedTensorId A map between the TensorId of the
   *                                      innerLoopSubgraph and the TensorId
   *                                      belonging to the cloned Ops in the
   *                                      fill stage. These tensors represent
   *                                      the input to the main pipeline cycle.
   * \return                              A map which can be used to connect
   *                                      tensors to the input of the
   *                                      flush phase
   */
  std::map<std::pair<PipelineStage, TensorId>, TensorId>
  modifyInputAndOutputInInnerLoop(
      const std::map<TensorId, TensorId> lastOriginalAndClonedTensorId);

  /**
   * Unroll the loop operator and create the flush phase.
   *
   * \param tensorIdsToFlushStage A map which can be used to connect tensors to
   *                              the input of the flush phase
   */
  void createFlushPhase(const std::map<std::pair<PipelineStage, TensorId>,
                                       TensorId> tensorIdsToFlushStage);

  /**
   * Remove the cloned graph and unset the pipeline stage on all operators.
   */
  void cleanUp();
};

} // namespace popart

#endif
