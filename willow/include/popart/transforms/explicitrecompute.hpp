// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXPLICITRECOMPUTE_HPP
#define GUARD_NEURALNET_EXPLICITRECOMPUTE_HPP

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <popart/graphutils.hpp>
#include <popart/transforms/transform.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/names.hpp"
#include "popart/pointercomparators.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class Graph;
class Op;

/**
 * Struct describing the context in which a Tensor or Op occurs.
 */
struct ExplicitRecomputeTensorContext {
  ExplicitRecomputeTensorContext(bool,
                                 OptionalExecutionPhase,
                                 OptionalPipelineStage);

  /**
   * Arbitrary sorting for containers
   * \param rhs Other ExplicitRecomputeTensorContext instance
   * \return    True if this < other
   */
  bool operator<(const ExplicitRecomputeTensorContext &rhs) const;

  bool operator==(const ExplicitRecomputeTensorContext &rhs) const;
  bool operator!=(const ExplicitRecomputeTensorContext &rhs) const;

  bool isForwardOp;
  OptionalExecutionPhase executionPhase;
  OptionalPipelineStage pipelineStage;
};

/**
 * Helper class to classify operations for recomputation by assigning an
 * ExplicitRecomputeTensorContext to each Op.
 * \ref ExplicitRecomputeTensorContext
 */
class ExplicitRecomputeHelper {
public:
  ExplicitRecomputeHelper(Graph &graph);

  /**
   * Get the context of an Op.
   * \param op Op to get the context for
   * \return   The context for the Op
   */
  ExplicitRecomputeTensorContext getContext(Op *op) const;

  /**
   * Get the contexts of all consumers of an Op's outputs.
   * \param op Op to get the output consumers' contexts for
   * \return   Contexts associated with all consumers.
   */
  std::set<ExplicitRecomputeTensorContext> getConsumerContexts(Op *op) const;

  /**
   * Check if this context is associated with the forward pass.
   * \param context Context to check
   * \return        True if the context belongs to the forward pass
   */
  bool isForwardContext(ExplicitRecomputeTensorContext context) const;

  /**
   * Register a new Op in the relation map
   * \param op      Op to register
   */
  void registerRecomputedOpRelation(Op *op);

  /**
   * Convert consumer contexts into valid recompute contexts. A valid recompute
   * context is the closest previous context that is able to recompute the
   * operation such that the consumer operation in the consumer context can
   * use the recomputed value.
   *
   * If no valid recompute context for a certain consumer context exists,
   * then an empty set is returned.
   *
   * \param  producerContext  Single context in which the original tensor is
   *                          produced.
   * \param  consumerContexts Set of consumer contexts to try to recompute for.
   * \return                  Set of valid recompute contexts. Can be empty.
   */
  std::set<ExplicitRecomputeTensorContext> getValidRecomputeContexts(
      const ExplicitRecomputeTensorContext &producerContext,
      const std::set<ExplicitRecomputeTensorContext> &consumerContexts) const;

  /**
   * Get the Op schedule of the graph state when the helper was created.
   * \return Vector of scheduled ops.
   */
  const std::vector<Op *> getOpSchedule() const { return schedule; }

  /**
   * Get the graph this helper is operating on.
   * \return The graph on which this helper operates on.
   */
  Graph &getGraph() const { return graph; }

  /**
   * Clone every recompute Op.
   */
  void cloneRecomputeOps();

  /**
   * Remap consumer Op inputs to use recomputed tensors where indicated
   * by matching contexts.
   */
  void remapConsumers();

private:
  Graph &graph;
  std::vector<Op *> schedule;
  std::map<std::pair<TensorId, ExplicitRecomputeTensorContext>, TensorId>
      recomputedTensorMap;
  std::map<PipelineStage, std::set<VGraphId>> pipelineStageVraphIdMap;
  std::map<Op *, graphutils::OpFinalLossRelation, POpCmp> relationMap;
};

/**
 * Explicit recomputation is a transformation that clones forward-pass
 * operations marked for recomputation and clones them.
 *
 * Consider a fragment of the training graph before the explicit recomputation
 * transform, where one gradient operation (CheckpointOp1Grad) requires a
 * value from the forward pass (RecomputeOp1) which is considered for
 * recomputation:
 *
 * CheckpointOp0
 *     |
 * RecomputeOp0
 *     |
 * RecomputeOp1 -.     ...
 *     |          \     |
 * CheckpointOp1    CheckpointOp1Grad
 *    ...              ...
 *     |                |
 *   Loss --------------
 *
 * (where CheckpointOp* is an op with
 * op->settings.recomputeType == RecomputeType::Checkpoint
 * and RecomputeOp* is an op with
 * op->settings.recomputeType == RecomputeType::Recompute)
 *
 * By marking these ops as 'recompute', the output of RecomputeOp1 does not
 * need to remain live until the recomputation of CheckpointOp1Grad. In other
 * words, the memory used to store this tensor is freed for allocation of other
 * tensors as soon as RecomputeOp1's output is read during the computation of
 * CheckpointOp1. How does this work in practice?
 *
 * After the transform, the graph fragment will look like:
 *
 * CheckpointOp0 -.
 *     |           \
 * RecomputeOp0   RecomputeOp0Clone
 *     |                  |
 * RecomputeOp1   RecomputeOp1Clone      ...
 *     |                  |               |
 * CheckpointOp1           ----- CheckpointOp1Grad
 *    ...                                ...
 *     |                                  |
 *   Loss --------------------------------
 *
 * Where every operation marked as `Recompute` will be cloned and added to
 * the backward pass, while all `Checkpoint` operation will remain connected
 * as-is.
 *
 * In pipelining, every copy operation between pipeline stages is (required to
 * be) checkpointed (in order to not cause data dependencies between stages
 * running in parallel), while everything else is recomputed. The user can
 * choose to checkpoint more, but not recompute more (with pipelining).
 *
 * The alternative, in the case of implicit recomputation, is to not transform
 * the graph at the IR level, and to use these recomputation settings to affect
 * the Ir lowering. In this case, the `poplar::program::Sequence`s that
 * correspond to the lowered RecomputeOps are added once to the main program as
 * scheduled in the forward pass, and then again directly preceding the
 * `poplar::program::Sequence` of the CheckpointOp1Grad. See the
 * `FindRequiredRecomputes class in irlowering.cpp
 *
 */
class ExplicitRecompute : public Transform {
public:
  static std::size_t id();

  ExplicitRecompute() : Transform() {}
  virtual ~ExplicitRecompute() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final {
    return "ExplicitRecompute";
  }
};

} // namespace popart

#endif
