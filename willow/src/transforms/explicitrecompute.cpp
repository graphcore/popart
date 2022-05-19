// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/operators.hpp"
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
#include <popart/op/identity.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
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

        // Register relationship between original and cloned operation
        recomputeMap[cloneOp] = op;

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

Tensor *ExplicitRecomputeHelper::getInplaceModifiedBackupTensor(
    Tensor *originalTensor,
    Op *originalOp,
    Op *recomputedOp,
    std::set<Op *, POpCmp> inplaceModifiers) {

  OpsBeforeKey keys;

  // Count modifiers after the loss, i.e. inplace modifying operations that have
  // a path from the loss and are thereby part of the backward pass
  unsigned numModifiersAfterLoss = 0;
  for (auto modifier : inplaceModifiers) {
    keys[modifier] = {recomputedOp};
    if (relationMap.at(modifier) == graphutils::OpFinalLossRelation::FromLoss) {
      numModifiersAfterLoss++;
    }
  }

  // We check if a recomputed operation can be scheduled early enough to avoid
  // consuming mangled values. If so, the inplace modified input tensor does not
  // need to be backed up. This could be improved, since this check
  // will force some recomputations to occur too early, rendering them less
  // useful or completely useless. Some additional tensors could be backed up to
  // allow recomputing later, although the heuristics for that are difficult.
  // Backup of tensors should not occur too aggressively either, i.e. it is
  // typically the case that weight updates occur after recomputation anyways.

  // Check if the graph can be scheduled such that the inplace modifiers occur
  // after the recomputed operation. However, if there are any modifiers that
  // do occur before the loss, i.e.
  // inplaceModifiers.size() != numModifiersAfterLoss
  // then we will backup the inplace modified tensor regardless, such that we
  // don't force the recomputation to occur before the inplace modifier in the
  // forward pass, and thereby before the loss, because ideally
  // recomputation should be scheduled as part of the backward pass in order
  // to minimize liveness
  if (graph.isSchedulable(keys) &&
      (inplaceModifiers.size() == numModifiersAfterLoss)) {
    // No issue to schedule modifiers after recompute, and no modifiers before
    // the loss
    return nullptr;
  }

  // Capture the state of the tensor by checking inplace modifying operations
  // before and after the original non-recomputed consumer of the tensor
  std::set<OpId> opsBefore;
  std::set<OpId> opsAfter;

  std::map<Op *, int, POpCmp> opToPosition;

  for (int i = 0; i < schedule.size(); ++i) {
    opToPosition[schedule.at(i)] = i;
  }

  int currentPos = -1;

  auto currentPosIt = opToPosition.find(originalOp);

  if (currentPosIt != opToPosition.end()) {
    currentPos = currentPosIt->second;
  }

  // Check if a modifier is scheduled before or after the original consumer
  // of the tensor
  for (auto modifier : inplaceModifiers) {
    auto modifierPosIt = opToPosition.find(modifier);
    if (modifierPosIt != opToPosition.end()) {
      auto modifierPos = modifierPosIt->second;
      if (currentPos <= modifierPos) {
        opsAfter.insert(modifier->id);
      } else if (currentPos > modifierPos) {
        opsBefore.insert(modifier->id);
      }
    }
  }

  // Check if a backup tensor already exists for the same state of the tensor
  // (i.e. same modifiers before and after the original consumer)
  if (inplaceModifiedBackupMap.find(
          {originalTensor->id, opsBefore, opsAfter}) ==
      inplaceModifiedBackupMap.end()) {

    // No new
    auto newId = graph.getIr().createIntermediateTensorId(originalTensor->id);
    IdentityOp *identityOp = graph.createConnectedOp<IdentityOp>(
        {{IdentityOp::getInIndex(), originalTensor->id}},
        {{IdentityOp::getOutIndex(), newId}},
        Onnx::Operators::Identity_1,
        originalOp->settings);

    // Ensure the IdentityOp that backs up the tensor before it is modified
    // occurs right before the original consumer, such that we capture the
    // tensor's values exactly as when the original operation consumes the
    // tensor. Note that the originalOp itself may modify the tensor too!
    graph.topoCons->insert(identityOp, originalOp, false);

    // Same modifiers before identityOp as before originalOp
    for (auto opId : opsBefore) {
      graph.topoCons->insert(graph.getOp(opId), identityOp, false);
    }

    // Same modifiers after identityOp as before originalOp
    for (auto opId : opsAfter) {
      graph.topoCons->insert(identityOp, graph.getOp(opId), false);
    }

    // Register backup tensor ID
    inplaceModifiedBackupMap[{originalTensor->id, opsBefore, opsAfter}] = newId;
  }

  // Return backup tensor ID
  return originalTensor->getGraph().getTensor(
      inplaceModifiedBackupMap.at({originalTensor->id, opsBefore, opsAfter}));
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

  // Remap inplace overwritten inputs such that recomputation results in the
  // same output as the original computation
  for (auto &op : graph.getOps()) {
    auto consumer = op.second.get();
    for (auto originalTensor : consumer->input->tensors()) {
      if (consumer->settings.recomputeType == RecomputeType::Recomputed) {
        auto inplaceModifiers = originalTensor->getInplaceModifiers();
        if (!inplaceModifiers.empty()) {

          auto backupTensor =
              getInplaceModifiedBackupTensor(originalTensor,
                                             recomputeMap.at(consumer),
                                             consumer,
                                             inplaceModifiers);

          if (backupTensor) {
            logging::transform::trace(
                "[ExplicitRecompute] Op {} has to consume backup tensor "
                "{}->{}.",
                consumer->debugName(),
                originalTensor->id,
                backupTensor->id);

            auto indices = consumer->input->indices(originalTensor);
            for (auto i : indices) {
              consumer->disconnectInTensor(i, originalTensor);
              consumer->connectInTensor(i, backupTensor->id);
            }
          }
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
