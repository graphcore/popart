// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PIPELINE_HPP
#define GUARD_NEURALNET_PIPELINE_HPP

#include <popart/graph.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/restore.hpp>
#include <popart/transforms/decomposeloops.hpp>
#include <popart/transforms/mainloops.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {

class Ir;
class LoopOp;

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
   * \param _batchesPerStep   Batches per step
   * \param _gradAcclFactor   Gradient accumulation factor
   * \param _maxPipelineStage The last pipeline stage
   * \param _doTraining       Optimizers are going to be applied (not inference)
   * \param _doGradAccl       Gradients are accumulated rather than applied to
   *                          the weights directly.
   */
  PipelineInfo(int64_t _batchesPerStep,
               int64_t _gradAcclFactor,
               int64_t _maxPipelineStage,
               bool _doTraining,
               bool _doGradAccl);

  int64_t numStages;

  bool doTraining;
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
   * Checks if a \a stage stage needs to be executed in a \a cycle
   * \param pCycle the cycle to check
   * \param pStage the stage to check
   * \return true  if a stage is to be executed in a cycle
   */
  bool doStage(PipelineCycle pCycle, PipelineStage pStage) const;

  /**
   * Number of cycles in the main phase interval
   * \return Number of cycles, start and end inclusive (closed interval)
   */
  PipelineCycle getMainCycles() const {
    return mainPhase.end - mainPhase.start + 1;
  }
};

/**
 * Struct holding tensors and operations required to set up stashes
 */
struct PipelineStashInfo {
  /**
   * TensorIds which are candidates for stashing.
   */
  std::set<TensorId> toStashTensors;

  /**
   * StashTensorId -> std::pair<StashRefOp, RestoreRefOps>
   * StashRefOp:    Operation that produces a tensor which is a candidate for
   *                stashing.
   * RestoreRefOps: Operations that are consumers of a tensor which is a
   *                candidate for stashing.
   */
  std::map<TensorId, std::pair<Op *, std::vector<Op *>>> stashRestoreRefOps;
};

/**
 * Struct holding metadata required to set up dynamic stash operations
 */
struct PipelineDynamicStashInfo {

  /**
   * The tensor to stash
   */
  Tensor *tensor;

  /**
   * Number of required stash entries
   * (stash tensor shape is [stashSize, *tensorShape])
   */
  int stashSize;

  /**
   * Single reference operation from which the DynamicUpdateOp (for stashing)
   * will inherit placement (IPU and pipeline stage)
   */
  Op *stashRefOp;

  /**
   * Vector of reference operations from which the DynamicSliceOp (for
   * restoring) will inherit placement (IPU and pipeline stage)
   */
  std::vector<Op *> restoreRefOps;

  /**
   * Vector of stages in which the tensor needs to be restored
   */
  std::vector<PipelineStage> restoreStages;

  /**
   * All consumers of the tensor (of which a subset will be required to consume
   * the restored tensor).
   */
  std::vector<Op *> consumers;

  /**
   * Base TensorId for the stash tensors
   */
  TensorId stashTensorBaseId;

  /**
   * Base TensorId for the stash counters
   */
  TensorId stashCounterBaseId;
};

class Pipeline : public Transform {
public:
  static std::size_t id();

  static bool checkIsFullRecompute(Graph &graph);

  Pipeline() : Transform() {}
  virtual ~Pipeline() override {}

  /**
   * Checks if the pipelining settings are valid and applies either
   * implicit or explicit pipelining transforms to the graph
   * \param graph top-level IR graph (main graph) for implicit pipelining,
   *              pipeline loop subgraph for explicit pipelining
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
   * Implicit pipelining and implicit recompute only!
   * This annotation pass will try to set the Ops between
   * the topologically final Checkpoints and the loss
   * to NOT be recomputed. This avoid a program where
   * operations are run twice in a row with no benefit to
   * liveness.
   * \param graph top-level IR graph (main graph)
   */
  static void setFinalFwdStageRecomputation(Graph &graph);

  /**
   * Check and adjust pipeline stage annotations on operations
   * \param graph Graph on which to check pipeline stages
   */
  static void checkOpsPipelineStage(Graph &graph);

  /**
   * Add all required dynamic update and dynamic slice operations to the graph,
   * which link forward and recompute/backward stages together via stashes
   * Only works for explicit pipelining
   * \param  graph Pipeline loop subgraph
   * \return       True if successful, will raise error if not
   */
  bool addDynamicStashAndRestoreOps(Graph &graph) const;

  /**
   * Add required IpuCopyOps to ensure that within the pipelined execution,
   * no copies between non-contiguous pipeline stages occur
   * \param  graph Pipeline loop subgraph
   * \return       True if successful, will raise error if not
   */
  bool contiguateIpuCopies(Graph &graph) const;

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
   * Select tensors candidates and prepare
   * (select reference operations for both stash and restore)
   * the IR for stashing
   * \param  graph                   Pipeline loop subgraph
   *                                 (explicit pipelining)
   *                                 or main graph (implicit pipelining)
   * \param  toStashCandidateTensors Candidates for stashing
   * \return                         Struct holding tensors and operations
   *                                 required to set up stashes
   */
  PipelineStashInfo
  prepareForStashing(Graph &graph,
                     std::set<TensorId> toStashCandidateTensors) const;

  /**
   * Create a stash tensor and add a DynamicUpdateInplaceOp to write a tensor
   * to the stash
   * \param graph  Pipeline loop subgraph
   * \param info   Metadata for creating the stash
   */
  TensorId
  addDynamicStashOpForTensor(Graph &graph,
                             const PipelineDynamicStashInfo &info) const;

  /**
   * Create a DynamicSliceOp to restore a tensor from the stash
   * \param graph                   Pipeline loop subgraph
   * \param info                    Metadata for creating the stash
   * \param restoreRefOpIndex       The index of the current restore operation
   * \param stashTensorInnerGraphId The TensorId of the stash tensor
   *                                (restore operation stash input)
   * \param lastRestoreTensorId     TensorId to use for the current restore
   *                                operation slice input
   * \return                        TensorId to use for the next restore
   *                                operation slice input
   *                                (current restore operation slice output)
   */
  TensorId addDynamicRestoreOpForTensor(Graph &graph,
                                        const PipelineDynamicStashInfo &info,
                                        size_t restoreRefOpIndex,
                                        TensorId stashTensorInnerGraphId,
                                        TensorId lastRestoreTensorId) const;

  /**
   * Create all required DynamicSliceOp to restore a tensor from the stash
   * (calls \ref addDynamicRestoreOpForTensor)
   * \param graph                   Pipeline loop subgraph
   * \param info                    Metadata for creating the stash
   * \param stashTensorInnerGraphId The TensorId of the stash tensor
   *                                (restore operation stash input)
   */
  void addDynamicRestoreOpsForTensor(Graph &graph,
                                     const PipelineDynamicStashInfo &info,
                                     TensorId stashTensorInnerGraphId) const;

  /**
   * Create all required stash and restore operations
   * (calls \ref addDynamicStashOpForTensor)
   * (calls \ref addDynamicRestoreOpsForTensor)
   * \param graph              Pipeline loop subgraph
   * \param pipelineStashInfo  Metadata for creating the stash
   * \param tid                TensorId of the tensor to stash
   */
  void addDynamicStashAndRestoreOpsForTensor(
      Graph &graph,
      const PipelineStashInfo &pipelineStashInfo,
      TensorId tid) const;

  /**
   * Add all required stash and restore operations to the graph, which link
   * forward and recompute/backward stages together
   * Only works for implicit pipelining
   * \param  graph Top-level IR graph (main graph)
   * \return       True if successful, will raise error if not
   */
  bool addStashRestoreOps(Graph &graph) const;

  /**
   * Transforms a linear set of pipeline stages into (explicit) pipeline cycles,
   * by inserting CallOps for each pipeline stage and unrolling the innermost
   * (explicit) loop partially
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
   * Map of all ops that should be outlined into CallOp
   */
  std::map<PipelineStage, std::vector<OpId>> opsToOutline;

  /**
   * Struct to help link \a OpIds to stages and stages to \a IpuCopyOps
   */
  struct PipelineStageOpIdMaps {
    std::map<OpId, PipelineStage> opIdAndPipelineStage;
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
   * Find operations that can be outlined in order to make the final, unrolled
   * pipeline graph more streamlined (such as reusing operations for each
   * pipeline stage).
   */
  void findOpsToOutline();

  /**
   * Create callOps of the operations picked in \ref findOpsToOutline.
   */
  void createCallOps();

  /**
   * Decompose the pipeline loop to represent explicit pipelining.
   */
  void decompose();

  /**
   * Unset the pipeline stage on all operators.
   */
  void cleanUp();

  /**
   * Calculate the minimum and maximum \c PipelineStage to unroll
   * \return Pair of {min, max} \c PipelineStages
   */
  std::pair<PipelineStage, PipelineStage> getMinAndMaxUnrollStages() const;
};

} // namespace popart

#endif
