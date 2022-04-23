// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/transforms/explicitrecompute.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/graphutils.hpp"
#include "popart/logging.hpp"
#include "popart/pointercomparators.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensornames.hpp"
#include "popart/transforms/transform.hpp"

namespace popart {

namespace {

auto getTensorContextTuple(const ExplicitRecomputeTensorContext ct) {
  return std::make_tuple(
      ct.isForwardOp,
      ct.executionPhase ? *ct.executionPhase : unusedExecutionPhase,
      ct.pipelineStage ? *ct.pipelineStage : unusedPipelineStage);
}

} // namespace

ExplicitRecomputeHelper::ExplicitRecomputeHelper(Graph &graph_)
    : graph(graph_) {
  relationMap = graphutils::getOpFinalLossRelations(graph_);
  schedule    = graph.getOpSchedule({}, RequireOptimalSchedule::No);

  for (Op *op : schedule) {
    if (op->hasPipelineStage() && op->hasVirtualGraphId()) {
      pipelineStageVGraphIdMap[op->getPipelineStage()].insert(
          op->getVirtualGraphId());
    }
  }
}

ExplicitRecomputeTensorContext::ExplicitRecomputeTensorContext(
    bool isForwardOp_,
    OptionalExecutionPhase executionPhase_,
    OptionalPipelineStage pipelineStage_)
    : isForwardOp(isForwardOp_), executionPhase(executionPhase_),
      pipelineStage(pipelineStage_) {}

bool ExplicitRecomputeTensorContext::
operator<(const ExplicitRecomputeTensorContext &rhs) const {
  return getTensorContextTuple(*this) < getTensorContextTuple(rhs);
}

bool ExplicitRecomputeTensorContext::
operator==(const ExplicitRecomputeTensorContext &rhs) const {
  return getTensorContextTuple(*this) == getTensorContextTuple(rhs);
}

bool ExplicitRecomputeTensorContext::
operator!=(const ExplicitRecomputeTensorContext &rhs) const {
  return getTensorContextTuple(*this) != getTensorContextTuple(rhs);
}

std::size_t ExplicitRecompute::id() {
  return typeid(ExplicitRecompute).hash_code();
}

ExplicitRecomputeTensorContext
ExplicitRecomputeHelper::getContext(Op *op) const {

  OptionalExecutionPhase executionPhase =
      (graph.getIr().getSessionOptions().executionPhaseSettings.phases > 1 &&
       op->hasExecutionPhase())
          ? op->getOptionalExecutionPhase()
          : OptionalExecutionPhase();
  OptionalPipelineStage pipelineStage =
      (graph.getIr().getSessionOptions().enablePipelining &&
       op->hasPipelineStage())
          ? op->getOptionalPipelineStage()
          : OptionalPipelineStage();

  // Recomputed operations should not count as forward operations, even though
  // they do not spawn from the loss
  bool isForwardOp =
      op->settings.recomputeType != RecomputeType::Recomputed &&
      (relationMap.at(op) == graphutils::OpFinalLossRelation::ToLoss ||
       relationMap.at(op) == graphutils::OpFinalLossRelation::FromToLoss);

  return ExplicitRecomputeTensorContext(
      isForwardOp, executionPhase, pipelineStage);
}

std::set<ExplicitRecomputeTensorContext>
ExplicitRecomputeHelper::getConsumerContexts(Op *op) const {
  std::set<ExplicitRecomputeTensorContext> contexts;

  for (auto &output : op->output->tensorMap()) {
    for (auto &consumer : output.second->consumers.getOps()) {
      contexts.insert(getContext(consumer));
    }
  }

  return contexts;
}

bool ExplicitRecomputeHelper::isForwardContext(
    ExplicitRecomputeTensorContext context) const {
  return context.isForwardOp;
}

void ExplicitRecomputeHelper::registerRecomputedOpRelation(Op *op) {
  relationMap[op] = graphutils::OpFinalLossRelation::ToFromLoss;
}

std::set<ExplicitRecomputeTensorContext>
ExplicitRecomputeHelper::getValidRecomputeContexts(
    const ExplicitRecomputeTensorContext &producerContext,
    const std::set<ExplicitRecomputeTensorContext> &consumerContexts) const {
  std::set<ExplicitRecomputeTensorContext> recomputeContexts;
  for (auto &consumerContext : consumerContexts) {

    if (producerContext.pipelineStage && consumerContext.pipelineStage) {
      auto producerVGrapIds =
          pipelineStageVGraphIdMap.at(*producerContext.pipelineStage);
      auto consumerVGrapIds =
          pipelineStageVGraphIdMap.at(*consumerContext.pipelineStage);
      std::set<VGraphId> intersection;
      std::set_intersection(producerVGrapIds.begin(),
                            producerVGrapIds.end(),
                            consumerVGrapIds.begin(),
                            consumerVGrapIds.end(),
                            std::inserter(intersection, intersection.end()));
      if (!intersection.empty()) {
        // Producer and consumer share a virtual graph ID - recomputation in the
        // context is possible. Multiple recompute is supported.
        recomputeContexts.insert(consumerContext);
      }
    } else if (producerContext.executionPhase &&
               consumerContext.executionPhase) {
      // Remap from forward to backward execution phase
      // Multiple recompute is not supported on phased execution yet.
      // There are no plans for adding support on this, since balancing
      // recompute with RemoteLoad/RemoteStore in phased execution would
      // require difficult heuristics.

      // Compute: backward_phase = 2 * number_of_phases - 2 - forward_phase
      ExplicitRecomputeTensorContext recomputeContext(
          false,
          2 * graph.getIr().getSessionOptions().executionPhaseSettings.phases -
              2 - *producerContext.executionPhase,
          OptionalPipelineStage());
      // Recompute needs to occur before the consumer context
      if (*recomputeContext.executionPhase <= *consumerContext.executionPhase) {
        recomputeContexts.insert(recomputeContext);
      }
    } else {
      // No pipelining or phased execution
      recomputeContexts.insert(consumerContext);
    }
  }
  return recomputeContexts;
}

void ExplicitRecomputeHelper::cloneRecomputeOps() {
  auto &schedule = getOpSchedule();

  for (auto opIt = schedule.rbegin(); opIt != schedule.rend(); ++opIt) {
    Op *op       = *opIt;
    auto context = getContext(op);

    if (op->settings.recomputeType == RecomputeType::Recompute) {

      auto producerContext  = getContext(op);
      auto consumerContexts = getConsumerContexts(op);

      // Change every recompute op to checkpoint
      op->settings.recomputeType = RecomputeType::Checkpoint;

      auto recomputeContexts =
          getValidRecomputeContexts(producerContext, consumerContexts);

      for (auto &recomputeContext : recomputeContexts) {
        // Recompute not required for context
        if (isForwardContext(recomputeContext) || recomputeContext == context) {
          continue;
        }

        // Recompute once per non-forward context
        auto cloneOpUp = op->clone();
        auto cloneOpId = getGraph().moveIntoGraph(std::move(cloneOpUp));

        Op *cloneOp = getGraph().getOp(cloneOpId);

        if (recomputeContext.executionPhase) {
          cloneOp->setExecutionPhase(*recomputeContext.executionPhase);
        }

        if (recomputeContext.pipelineStage) {
          cloneOp->setPipelineStage(*recomputeContext.pipelineStage);
        }

        cloneOp->disconnectAllInputs();
        cloneOp->disconnectAllOutputs();
        cloneOp->settings.recomputeType = RecomputeType::Recomputed;

        registerRecomputedOpRelation(cloneOp);

        for (auto &in : op->input->tensorMap()) {
          // Consume forward produced tensor initially
          cloneOp->connectInTensor(in.first, in.second->id);
        }
        for (auto &out : op->output->tensorMap()) {
          TensorId recomputedId = op->getIr().createIntermediateTensorId(
              createRecomputedTensorId(out.second->id));
          recomputedTensorMap[{out.second->id, recomputeContext}] =
              recomputedId;
          cloneOp->createAndConnectOutTensor(out.first, recomputedId);
        }
        cloneOp->setup();

        logging::transform::trace("[ExplicitRecompute] Cloned op {} -> {}",
                                  op->debugName(),
                                  cloneOp->debugName());
      }
    }
  }
}

void ExplicitRecomputeHelper::remapConsumers() {
  for (auto recomputedTensor : recomputedTensorMap) {
    Tensor *originalTensor =
        graph.getTensors().get(recomputedTensor.first.first);
    auto producer = originalTensor->getProducer();
    for (Op *consumer : originalTensor->consumers.getOps()) {
      auto producerContext = getContext(producer);
      auto consumerContext = getContext(consumer);
      auto recomputeContexts =
          getValidRecomputeContexts(producerContext, {consumerContext});
      for (auto recomputeContext : recomputeContexts) {
        if (recomputeContext == recomputedTensor.first.second) {
          auto indices = consumer->input->indices(originalTensor);
          for (auto i : indices) {
            consumer->disconnectInTensor(i, originalTensor);
            consumer->connectInTensor(i, recomputedTensor.second);
          }
          logging::transform::trace(
              "[ExplicitRecompute] Op {} consumes recomputed tensor {}->{}",
              consumer->debugName(),
              originalTensor->id,
              recomputedTensor.second);
        } else {
          logging::transform::trace(
              "[ExplicitRecompute] Op {} cannot consume recomputed tensor "
              "{}->{}.",
              consumer->debugName(),
              originalTensor->id,
              recomputedTensor.second);
        }
      }
    }
  }
}

bool ExplicitRecompute::apply(Graph &graph) const {
  logging::transform::debug("[ExplicitRecompute] Started.");

  ExplicitRecomputeHelper helper(graph);

  // Clone every recompute Op
  helper.cloneRecomputeOps();

  // Remap consumer Op inputs to use recomputed tensors where indicated
  // by matching contexts
  helper.remapConsumers();

  logging::transform::debug("[ExplicitRecompute] Done.");
  return true;
}

namespace {
// ExplicitRecompute
bool init = Transform::registerTransform(new ExplicitRecompute());
} // namespace

} // namespace popart
