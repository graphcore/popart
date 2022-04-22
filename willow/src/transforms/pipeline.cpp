// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>
#include <popart/aliasesmap.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/incrementmod.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/patterns/contiguateipucopyindices.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/mainloops.hpp>
#include <popart/transforms/overlapio.hpp>
#include <popart/transforms/pipeline.hpp>
#include <popart/transforms/randomsetup.hpp>
#include <popart/transforms/subgraphoutline.hpp>
#include <popart/util.hpp>
#include <popart/vertex.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/dataflow.hpp"
#include "popart/datatype.hpp"
#include "popart/logging.hpp"
#include "popart/op/exchange/exchange.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/pointercomparators.hpp"
#include "popart/region.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/tensornames.hpp"
#include "popart/transforms/decomposeloops.hpp"
#include "popart/transforms/transform.hpp"
#include "popart/vendored/optional.hpp"

// Which pipelining scheme should we use? There are some considerations to
// make:
//  - Which order should the 5 progs (Fwd, Bwd, Stash, Restore, Sync)
//    run in the pipeline cycle?
//  - Should the activation be restored in-place?
//
// These decisions affect the stash sizes and the total number of activations
// that have to be stored in the pipelined model.
//
// We have decided on:
//  - In-place the activation tensors when restoring, when possible
//  - Running : Fwd/Stash/Restore/Bwd/Sync
// resulting in a stash size of 2*(IPUs to end) + 1, except for the last
// IPU, which has no stash.
//
// You can get away with a smaller stash, at the expense of having to out-place
// activations and changing the program order. Our approach is conceptually
// straight-forward and has no memory penalty.
//
// The transform
// -------------
//
// Before:
//
// FwdOp     t_act_grad
//   |          |
//  t_act --- BwdOp
//   |          |
//  ...      t_grad_in
//
// After:
//
// FwdOp
//   |
// t_act ----------           t_act_grad
//   | \           |             |
//   |   \      StashOp          |
//   |     \       |             |
//   |       \   t_act_stashed   |
//   |        |    |             |
//   |        |    |             |
//   |     RestoreOp             |
//   |       |                   |
//   |     t_act_alias ------- BwdOp
//   |                           |
//   |                       t_grad_in
//  ...

namespace popart {

std::size_t Pipeline::id() { return typeid(Pipeline).hash_code(); }

namespace {

VGraphId getVirtualGraphIdOrSourceIpu(Op *op, Tensor *t) {
  auto consumers = t->consumers.getOps();
  if (t->getProducerUnsafe() == op) {
    return op->getIntrospectionOutVirtualGraphId(op->output->indices(t).front())
        .first;
  } else if (std::find(consumers.begin(), consumers.end(), op) !=
             consumers.end()) {
    return op->getIntrospectionInVirtualGraphId(op->input->indices(t).front())
        .first;
  }
  return op->getVirtualGraphId();
}

void setIPUCopyPipelineStage(IpuCopyOp *op) {

  auto in0 = op->inTensor(0);
  if (in0->hasProducer()) {
    auto producer = in0->getProducer();
    auto ps       = producer->getPipelineStage();
    op->setPipelineStage(ps);
  } else {
    if (in0->tensorType() == TensorType::Variable) {
      throw error("Can not copy variable tensor {} between virtual graphs when "
                  "pipelining. All pipeline stages using this tensor should be "
                  "on the same graph.",
                  in0->str());
    } else {
      // Const or Stream tensors
      auto ps = in0->consumers.findLowestPipelineStage();
      op->setPipelineStage(ps);
    }
  }
}

Op *getStashReferenceOp(Tensor *t) {
  // Choose an op for the stash op to copy the vgraph and pstage from.

  // If the tensor has no producer, or the producer is a copy op, then the
  // tensor has been streamed/copied onto this virtual graph just in time to be
  // consumed. There must also be a later consumer on the same virtual graph,
  // otherwise this tensor would not have been a candidate for stashing. Use the
  // consumer with the lowest pipeline stage as the stash ref op.
  if (!t->hasProducer() || t->getProducer()->isConvertibleTo<IpuCopyOp>()) {
    auto consumers  = t->consumers.getOps();
    auto stashRefOp = consumers.at(0);
    for (auto c : t->consumers.getOps()) {
      if (c->getPipelineStage() < stashRefOp->getPipelineStage()) {
        stashRefOp = c;
      }
    }

    return stashRefOp;
  }
  // The tensor has been produced by an op on this virtual graph, and is to be
  // consumed by an op on this virtual graph in a later pipeline stage.
  else {
    return t->getProducer();
  }
}

std::string zeroCandidatesError(Tensor *t, Op *stashRefOp) {
  std::stringstream ss;
  ss << "No candidates for restore op.";

  ss << logging::format("\nTensor: {}", t->id);
  if (t->hasProducer()) {
    auto prod = t->getProducer();
    ss << logging::format("\n  Producer: {}, ps: {}, vg: {}",
                          prod->debugName(),
                          prod->getPipelineStage(),
                          getVirtualGraphIdOrSourceIpu(prod, t));
  }
  ss << "\n  Consumers:";
  for (auto c : t->consumers.getOps()) {
    ss << logging::format("\n    {}, ps: {}, vg: {}",
                          c->debugName(),
                          c->getPipelineStage(),
                          getVirtualGraphIdOrSourceIpu(c, t));
  }

  ss << logging::format("\nStash Ref Op: {}, ps: {}, vg: {}",
                        stashRefOp->debugName(),
                        stashRefOp->getPipelineStage(),
                        getVirtualGraphIdOrSourceIpu(stashRefOp, t));

  return ss.str();
}

// Find descendent consumers on different PipelineStages that can be used
// as restore reference Ops by searching through the consumers but not
// crossing IPU or executionContext boundaries.
std::vector<Op *> findDescendentsOnDifferentPipelineStages(Tensor *t,
                                                           Op *stashRefOp) {
  OpSearchHelper toCheck;

  std::vector<Op *> differentPipelineStageDescendents;
  std::vector<PipelineStage> foundPipelineStages;
  toCheck.pushConsumers(t);
  while (!toCheck.empty()) {
    auto op = toCheck.pop();
    if (op->isConvertibleTo<IpuCopyOp>()) {
      // do nothing
    } else if (op->getOptionalVGraphId() != stashRefOp->getOptionalVGraphId()) {
      // e.g. not a IpuCopyOp, but is an op that spans multiple VirtualGraphs,
      // such a collective op - do nothing
    } else if (op->settings.executionContext !=
               stashRefOp->settings.executionContext) {
      // do nothing
    } else if (op->getPipelineStage() > stashRefOp->getPipelineStage()) {
      if (std::find(foundPipelineStages.begin(),
                    foundPipelineStages.end(),
                    op->getPipelineStage()) == foundPipelineStages.end()) {
        differentPipelineStageDescendents.push_back(op);
        foundPipelineStages.push_back(op->getPipelineStage());
      }
    } else {
      toCheck.pushOutputConsumers(op);
    }
  }

  return differentPipelineStageDescendents;
}

std::vector<Op *> findImplicitRecomputeDependants(Op *restoreOp) {
  OpSearchHelper toCheck;

  std::vector<Op *> dependants;
  toCheck.pushConsumers(
      restoreOp->inTensor(RestoreInplaceOp::getActToRestoreInIndex()));
  while (!toCheck.empty()) {
    auto op = toCheck.pop();
    if (op->isConvertibleTo<IpuCopyOp>()) {
      // do nothing
    } else if (op->getOptionalVGraphId() != restoreOp->getOptionalVGraphId()) {
      // e.g. not a IpuCopyOp, but is an op that spans multiple VirtualGraphs,
      // such a collective op - do nothing
    } else if (op->settings.executionContext !=
               restoreOp->settings.executionContext) {
      // do nothing
    } else if (op->settings.recomputeType != RecomputeType::Recompute) {
      // Only follow sequences of Recompute operations.
      if (op->getPipelineStage() == restoreOp->getPipelineStage()) {
        dependants.push_back(op);
      }
    } else {
      toCheck.pushOutputConsumers(op);
    }
  }

  return dependants;
}

bool isProducedOnIPU(Tensor *tensor) {
  // Has a producer and it's a copy
  if (tensor->hasProducer() && tensor->getProducer()->isIpuCopyOp()) {
    return false;
  }
  if (tensor->hasProducer() &&
      tensor->getProducer()->isConvertibleTo<InitOp>()) {
    return true;
  }
  if (tensor->isHostLoadTensor()) {
    return false;
  }
  // Doesn't have a producer and it's a stream
  if (!tensor->hasProducer() && tensor->tensorType() == TensorType::Stream) {
    return false;
  }

  return true;
}

std::unique_ptr<IdentityOp> createIdenityCopyOp(Graph &graph,
                                                Tensor *tensor,
                                                VGraphId vGraphId,
                                                PipelineStage pStage) {
  Op::Settings identitySettings(graph, tensor->id + "_pipelineCopy");
  auto op = std::make_unique<IdentityOp>(Onnx::Operators::Identity_1,
                                         identitySettings);
  if (op == nullptr) {
    throw error("Failed to create op Identity Copy op, {}",
                Onnx::Operators::Identity_1);
  }

  // ensure that op is not inplaced
  op->settings.excludePatterns.insert({"InPlace", "ViewSimplifyPattern"});

  op->setVirtualGraphId(vGraphId);
  op->setPipelineStage(pStage);
  return op;
}

void insertClonesBeforeIpuCopyConsumers(Graph &graph,
                                        Tensor *tensor,
                                        VGraphId src_ipu,
                                        PipelineStage pStage) {
  // Modify the IR as shown below.
  //
  // Before:
  //
  //  Producer
  //     |
  //    tensor -------
  //     |    \       \
  //     |     \       \
  //     |      \       \
  //     |       \       \
  // Consumer  IpuCopy0  IpuCopy1
  //
  // After:
  //
  //  Producer
  //     |
  //    tensor
  //     |    \
  //     |   Identity
  //     |      \
  //     |     tenor_copy --
  //     |        \         \
  // Consumers   IpuCopy0  IpuCopy1

  std::vector<IpuCopyOp *> ipuCopyConsumers;
  for (auto *c : tensor->consumers.getOps()) {
    auto ipuCopyOp = dynamic_cast<IpuCopyOp *>(c);
    if (ipuCopyOp) {
      ipuCopyConsumers.push_back(ipuCopyOp);
    }
  }

  if (ipuCopyConsumers.size() > 0) {
    auto identity = createIdenityCopyOp(graph, tensor, src_ipu, pStage);
    TensorId identityOutput = identity->settings.name;
    identity->connectInTensor(0, tensor->id);
    identity->createAndConnectOutTensor(0, identityOutput);
    identity->setup();
    graph.moveIntoGraph(std::move(identity));

    for (auto *ipuCopyOp : ipuCopyConsumers) {
      for (auto inIndex : ipuCopyOp->input->indices(tensor)) {
        ipuCopyOp->disconnectInTensor(inIndex, tensor);
        ipuCopyOp->connectInTensor(inIndex, identityOutput, src_ipu);
      }
    }
  }
}

void insertCloneBeforeCopiesToHost(Graph &graph,
                                   Tensor *tensor,
                                   VGraphId vGraphId,
                                   PipelineStage pStage) {
  // Since we can't rename anchors (as the user is expecting anchor of
  // known name), we modify the IR as shown below.
  //
  // Before:
  //
  //  Producer
  //     |
  //  anchor_id
  //     |    \
  //     |     \
  //     |      \
  //     |       \
  // Consumers  ToHostStream
  //

  // Intermediate1:
  //
  //  Producer
  //     |
  //  new_tensor_id
  //     |
  //   Identity
  //
  //      anchor_id
  //       /      \
  // Consumers   ToHostStream

  // Final:
  //
  //  Producer
  //     |
  //  new_tensor_id
  //     |     \
  //     |    Identity
  //     |       \
  //     |       anchor_id
  //     |         \
  // Consumers   ToHostStream

  // Intermediate1
  auto producer = tensor->getProducer();
  auto outIndex = producer->outIndex(tensor);
  producer->disconnectOutTensor(tensor);
  TensorId substituteId = tensor->id + "_substitute";
  producer->createAndConnectOutTensor(outIndex, substituteId);
  producer->setup();
  auto identity = createIdenityCopyOp(graph, tensor, vGraphId, pStage);
  identity->connectInTensor(0, substituteId);

  // Final
  identity->connectOutTensor(0, tensor->id);
  identity->setup();
  graph.moveIntoGraph(std::move(identity));
  // for each consumer of anchor tensor, disconnect and reconnect at same
  // indices to its substitute
  for (auto op : tensor->consumers.getOps()) {
    for (auto inIndex : op->input->indices(tensor)) {
      op->disconnectInTensor(inIndex, tensor);
      op->connectInTensor(inIndex, substituteId);
    }
  }
}

std::vector<Op *> getRestoreReferenceOps(Tensor *t, Op *stashRefOp) {
  logging::debug("Collecting restore ref candidates");
  auto consumers = t->consumers.getOps();

  std::vector<Op *> restoreCandidates;
  std::vector<PipelineStage> restorePipelineStages;
  for (auto c : consumers) {
    if (getVirtualGraphIdOrSourceIpu(c, t) ==
            getVirtualGraphIdOrSourceIpu(stashRefOp, t) &&
        c->getPipelineStage() != stashRefOp->getPipelineStage()) {
      if (c->getPipelineStage() < stashRefOp->getPipelineStage()) {
        throw internal_error(
            "All 'restore' reference ops must have a PipelineStage greater "
            "than that of the tensor they are restoring. But (PS of 'restore' "
            "reference candidate) {} < {} ( PS of 'stash' reference candidate)",
            c->getPipelineStage(),
            stashRefOp->getPipelineStage());
      } else {
        if (std::find(restorePipelineStages.begin(),
                      restorePipelineStages.end(),
                      c->getPipelineStage()) == restorePipelineStages.end()) {
          restoreCandidates.push_back(c);
          restorePipelineStages.push_back(c->getPipelineStage());
        }
      }
    }
  }

  if (restoreCandidates.size() == 0) {
    throw internal_error(zeroCandidatesError(t, stashRefOp));
  }

  return restoreCandidates;
}

// clang-format off
// (a) -> [copy] -> (a_copy0 on pipeline stage N)
// (a) -> [copy] -> (a_copy1 on pipeline stage M)
//                     ==================>
// (a) -> [copy] -> (a_copy0 on pipeline stage N) -> [copy] -> (a_copy1 on pipeline stage M)
// clang-format on
void chainCopies(std::vector<IpuCopyOp *> &copies) {
  for (auto c : copies) {
    if (c->input->n() > 1) {
      // Chaining copies with more than 1 input is possible, but I don't think
      // it will ever occur.
      throw internal_error(
          "Attempting to chain a copy with more than one input.");
    }
  }

  std::sort(copies.begin(), copies.end(), [](auto &lhs, auto &rhs) {
    auto lhsStage = *lhs->outTensor(0)->consumers.findLowestPipelineStage();
    auto rhsStage = *rhs->outTensor(0)->consumers.findLowestPipelineStage();
    return lhsStage < rhsStage;
  });

  for (int i = 1; i < copies.size(); i++) {
    auto prevCopyOp = copies[i - 1];
    auto copyOp     = copies[i];
    auto newPStage =
        prevCopyOp->outTensor(0)->consumers.findLowestPipelineStage();

    copyOp->disconnectInTensor(0, copyOp->inTensor(0));
    copyOp->connectInTensor(0, prevCopyOp->outId(0), prevCopyOp->getDestIpu());
    copyOp->setPipelineStage(newPStage);
  }
}

// Look for and transform groups of copies that may be chained. This prevents
// duplicate copies being created by the contiguate copies transform where:
//   O -> N
//   O -> M
// would become:
//   O -> O+1 -> O+2 -> ... -> N
//   O -> O+1 -> O+2 -> ... -> N -> N+1 -> N+2 -> ... -> M
void chainCopiesTransform(Graph &graph) {
  std::map<TensorId, std::vector<IpuCopyOp *>> copyMap;
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->isConvertibleTo<IpuCopyOp>() &&
        op->settings.executionContext == ExecutionContext::Normal) {
      auto copyOp = dynamic_cast<IpuCopyOp *>(op);
      for (auto tensor : op->input->tensors()) {
        copyMap[tensor->id].push_back(copyOp);
      }
    }
  }

  for (auto &tenId_copies : copyMap) {
    auto &copies = tenId_copies.second;

    if (copies.size() > 1) {
      chainCopies(copies);
    }
  }
}

void mergeConsecutivePipelineStages(Graph &graph) {
  if (graph.getIr()
          .getSessionOptions()
          .createImplicitPipeliningFwdOnlyProgram) {
    // Do not re-merge last forward and first backward stage
    return;
  }

  std::map<PipelineStage, VGraphId> stageMap;
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->hasPipelineStage() && op->hasVirtualGraphId() &&
        !op->isIpuCopyOp()) {
      stageMap[op->getPipelineStage()] = op->getVirtualGraphId();
    }
  }

  std::map<PipelineStage, PipelineStage> stageTransformation;

  unsigned shift = 0;
  VGraphId prev  = stageMap.begin()->second;
  for (auto &pStage_vGraphId : stageMap) {
    auto pStage   = pStage_vGraphId.first;
    auto vGraphId = pStage_vGraphId.second;
    if (pStage == 0) {
      continue;
    }
    if (vGraphId == prev) {
      logging::debug("Merging Pipeline Stage {} into Previous", pStage);
      shift++;
    }
    stageTransformation[pStage] = pStage - shift;
    prev                        = vGraphId;
  }

  // Apply stageTransformation
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->hasPipelineStage()) {
      op->setPipelineStage(stageTransformation[op->getPipelineStage()]);
    }
  }
}

bool hasCheckpointProducer(Tensor *tensor) {
  return !tensor->hasProducer() ||
         tensor->getProducer()->settings.recomputeType ==
             RecomputeType::Checkpoint;
}

bool onlyConsumedByPostLossOps(
    Tensor *tensor,
    std::map<Op *, graphutils::OpFinalLossRelation, POpCmp> relations) {
  if (tensor->getIr().getSessionOptions().explicitPipeliningEnabled()) {
    // Explicit code path
    for (auto consumer : tensor->consumers.getOps()) {
      if (relations.at(consumer) == graphutils::OpFinalLossRelation::ToLoss ||
          relations.at(consumer) ==
              graphutils::OpFinalLossRelation::FromToLoss) {
        return false;
      }
    }
    return true;
  } else {
    // Implicit code path
    for (auto consumer : tensor->consumers.getOps()) {
      if (consumer->scheduledPreLoss == ScheduledPreLoss::Yes) {
        return false;
      }
    }
    return true;
  }
}

std::set<TensorId> getStashCandidateTensors(Graph &graph) {
  bool isExplicit =
      graph.getIr().getSessionOptions().explicitPipeliningEnabled();

  bool isFullRecompute = Pipeline::checkIsFullRecompute(graph);

  std::map<Op *, graphutils::OpFinalLossRelation, POpCmp> relations;

  if (isExplicit) {
    relations = graphutils::getOpFinalLossRelations(graph);
  }

  std::set<TensorId> toStashCandidateTensors;
  for (auto &tid : graph.getTensors().getAllTensorIds()) {
    auto tensor = graph.getTensors().get(tid);

    bool stashing = true;
    std::string reason;

    if (tensor->consumers.getOps().empty()) {
      reason   = "the tensor not having any consumers.";
      stashing = false;
    }

    if (stashing && (tensor->tensorType() == TensorType::Variable ||
                     tensor->tensorType() == TensorType::Const)) {
      reason   = logging::format("the tensor being a {}", tensor->tensorType());
      stashing = false;
    }

    if (stashing && ((tensor->isGraphInput() && !isExplicit) ||
                     tensor->isImplicitLoopInput())) {
      reason = "the tensor being a non-stashable graph input (the value of "
               "implicit loop inputs remains either unchanged, or changes "
               "inplace, for all loop iterations, and therefore it is not "
               "required to back-up (stash) past values)";
      stashing = false;
    }

    if (stashing && tensor->isOptimizerTensor()) {
      reason = "the tensor being an optimizer tensor (the value of optimizer "
               "tensors remains unchanged across loop iterations)";
      stashing = false;
    }

    // Full recompute uses stashes only on the inputs to an IPU
    // to complete any pipeline stage, or tensors specified by the user.
    if (stashing && isFullRecompute && isProducedOnIPU(tensor) &&
        !hasCheckpointProducer(tensor)) {
      reason = logging::format(
          "the tensor not being an input to an IPU, and using full recompute "
          "(produced on IPU: {} checkpoint producer: {}) (the tensor will be "
          "recomputed instead of stashed and restored)",
          isProducedOnIPU(tensor),
          hasCheckpointProducer(tensor));
      stashing = false;
    }

    // We only concern ourselves with the normal and subgraphs context
    if (stashing && tensor->hasProducer() &&
        tensor->getProducer()->settings.executionContext !=
            popart::ExecutionContext::Normal &&
        tensor->getProducer()->settings.executionContext !=
            popart::ExecutionContext::Subgraph) {
      reason =
          logging::format("the tensor produced in context {}, which is not "
                          "part of a pipelined execution",
                          tensor->getProducer()->settings.executionContext);
      stashing = false;
    }

    auto onlyConsumedByCopies = [](Tensor *t) {
      for (auto consumer : t->consumers.getOps()) {
        if (!consumer->isConvertibleTo<IpuCopyOp>()) {
          return false;
        }
      }
      return true;
    };

    // Get all the stages the tensor is produced/consumed in.
    std::set<PipelineStage> tensorStages = tensor->getPipelineStages();

    // There is no need to stash a tensor that only appears in 1 stage.
    // Unless using full recompute, then it must be consumed by something
    // other than a copy (it's not just "passing through"), which is
    // scheduled pre-loss (we're not recomputing grad ops)
    if (stashing && tensorStages.size() == 1 &&
        !(!isExplicit && isFullRecompute && !onlyConsumedByCopies(tensor) &&
          !onlyConsumedByPostLossOps(tensor, relations))) {
      reason   = logging::format("only appearing in 1 stage: {} (there are no "
                               "further stages requiring the restored value)",
                               tensorStages);
      stashing = false;
    }

    if (stashing) {
      logging::transform::debug(
          "[getStashCandidateTensors] Adding {} to stash candidates", tid);
      toStashCandidateTensors.insert(tid);
    } else {
      logging::transform::trace(
          "[getStashCandidateTensors] Not stashing tensor {} due to {}.",
          tid,
          reason);
    }
  }

  return toStashCandidateTensors;
}

// Implicit recompute only, does not make use of the op->canRecompute() and
// has it's own rules instead.
bool isRecomputable(Op *op) {
  if (op->settings.executionContext != ExecutionContext::Normal &&
      op->settings.executionContext != ExecutionContext::Subgraph) {
    return false;
  }
  if (op->isConvertibleTo<HostLoadOp>()) {
    return false;
  }
  if (op->isConvertibleTo<HostStoreOp>()) {
    return false;
  }
  if (op->isConvertibleTo<InitOp>()) {
    return false;
  }

  // Copy ops are never recomputable
  if (op->isConvertibleTo<IpuCopyOp>()) {
    return false;
  }
  // Don't recompute the GetRandomSeedOp, or the identity that clones it.
  auto clonesRandomSeed = [&] {
    if (op->isConvertibleTo<IdentityOp>()) {
      auto input = op->inTensor(0);
      return input->hasProducer() &&
             input->getProducer()->isConvertibleTo<GetRandomSeedOp>();
    }
    return false;
  };
  if (op->isConvertibleTo<GetRandomSeedOp>() || clonesRandomSeed()) {
    return false;
  }

  return true;
}

bool outputsAreStashed(Op *op, std::set<TensorId> &stashTensors) {
  for (const auto &outT : op->output->tensors()) {
    if (stashTensors.find(outT->id) != stashTensors.end()) {
      return true;
    }
  }
  return false;
}

void setRecomputation(Graph &graph,
                      std::set<TensorId> &toStashCandidateTensors) {
  bool isFullRecompute = Pipeline::checkIsFullRecompute(graph);

  auto isConsumedByCopy = [](Op *op) {
    for (auto tensor : op->output->tensors()) {
      for (auto consumer : tensor->consumers.getOps()) {
        if (consumer->isConvertibleTo<IpuCopyOp>()) {
          return true;
        }
      }
    }
    return false;
  };

  auto isConsumedByHostLoad = [](Op *op) {
    for (auto tensor : op->output->tensors()) {
      for (auto consumer : tensor->consumers.getOps()) {
        if (consumer->isConvertibleTo<HostLoadOp>()) {
          return true;
        }
        if (MultiExchangeOp *multiExchangeOp =
                dynamic_cast<MultiExchangeOp *>(consumer)) {
          auto inIndex    = multiExchangeOp->input->indices(tensor).front();
          auto descriptor = multiExchangeOp->getExchangeDescriptor(
              multiExchangeOp->outIndexToDescriptorIndex(inIndex).first);
          if (descriptor.isHostExchange() &&
              descriptor.getDirection() == ExchangeDirection::Load) {
            return true;
          }
        }
      }
    }
    return false;
  };

  // Initialise ops to be Recompute, except Ops whose output enters an IpuCopy,
  // or IpuCopys themselves.
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (isRecomputable(op)) {
      if (isFullRecompute) {
        // In isFullRecompute all forward ops are Recomputed unless specified by
        // the user.
        if (op->settings.recomputeType != RecomputeType::Checkpoint &&
            !outputsAreStashed(op, toStashCandidateTensors) &&
            op->scheduledPreLoss == ScheduledPreLoss::Yes) {
          op->settings.recomputeType = RecomputeType::Recompute;
        }
      } else {
        if (isConsumedByCopy(op) || isConsumedByHostLoad(op)) {
          op->settings.recomputeType = RecomputeType::Checkpoint;
        } else {
          op->settings.recomputeType = RecomputeType::Recompute;
        }
      }
    } else if (op->isConvertibleTo<IpuCopyOp>()) {
      op->settings.recomputeType = RecomputeType::Checkpoint;
    }
  }

  // Finding initial set of Tensors which are not produced on their IPUs and
  // are not stashed
  TensorSearchHelper frontier;
  auto isStashCandidate = [&](Tensor *t) {
    return std::find(toStashCandidateTensors.cbegin(),
                     toStashCandidateTensors.cend(),
                     t->id) != toStashCandidateTensors.cend();
  };

  for (auto tid : graph.getTensors().getAllTensorIds()) {
    Tensor *tensor = graph.getTensors().get(tid);
    if (!isProducedOnIPU(tensor) && !isStashCandidate(tensor)) {
      frontier.push(tensor);
    }
  }

  // Starting from the initial frontier found above,
  // propogate "Checkpoint" forward til either a Stash Tensor or an IPU copy
  // is reached.
  while (!frontier.empty()) {
    Tensor *tensor = frontier.pop();
    for (Op *consumer : tensor->consumers.getOps()) {
      consumer->settings.recomputeType = RecomputeType::Checkpoint;
      if (!dynamic_cast<IpuCopyOp *>(consumer)) {
        for (Tensor *consumerOut : consumer->output->tensors()) {
          if (!isStashCandidate(consumerOut)) {
            frontier.push(consumerOut);
          }
        }
      }
    }
  }
}

GetRandomSeedOp *findGetRandomSeedOp(Graph &graph) {
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (auto x = dynamic_cast<GetRandomSeedOp *>(op)) {
      return x;
    }
  }
  throw error("Could not find an instance of GetRandomSeedOp in graph");
}

TensorId createStashableRandomSeed(GetRandomSeedOp *randomSeedOp) {
  auto randomSeed =
      randomSeedOp->outTensor(GetRandomSeedOp::getUpdatedSeedOutIndex());

  // Create the identity op to clone the random seed.
  logging::transform::debug("Adding Identity Copy for random seed tensor {}",
                            randomSeed->id);
  Op::Settings identitySettings(randomSeedOp->getGraph(),
                                randomSeed->id + "_pipelineCopyOp");
  TensorId identityOutput = randomSeed->id + "_pipelineCopy";
  IdentityOp *identityOp  = [&] {
    auto x = std::make_unique<IdentityOp>(Onnx::Operators::Identity_1,
                                          identitySettings);
    // ensure that op is not inplaced
    x->settings.excludePatterns.insert({"InPlace", "ViewSimplifyPattern"});
    auto op = x.get();
    randomSeedOp->getGraph().moveIntoGraph(std::move(x));
    return op;
  }();

  identityOp->connectInTensor(IdentityOp::getInIndex(), randomSeed->id);
  identityOp->createAndConnectOutTensor(IdentityOp::getOutIndex(),
                                        identityOutput);
  identityOp->setVirtualGraphId(randomSeedOp->getVirtualGraphId());
  identityOp->setPipelineStage(randomSeedOp->getPipelineStage());
  identityOp->setup();

  auto randomSeedClone = identityOp->outTensor(0);
  // Connect the consumers of random seed to the output of the identity op
  for (auto consumer : randomSeed->consumers.getOps()) {
    if (consumer != identityOp) {
      // Important to copy tensorMap here, as TensorIndex::tensorMap returns a
      // reference and we shall be modifiying it.
      auto inputMap = consumer->input->tensorMap();
      for (auto idx_tensor : inputMap) {
        auto idx         = idx_tensor.first;
        auto inputTensor = idx_tensor.second;
        if (inputTensor == randomSeed) {
          if (auto copyOp = dynamic_cast<IpuCopyOp *>(consumer)) {
            auto sourceIpu = copyOp->getSourceIpu();
            copyOp->disconnectInTensor(idx, randomSeed);
            copyOp->connectInTensor(idx, randomSeedClone->id, sourceIpu);
          } else {
            consumer->disconnectInTensor(idx, randomSeed);
            consumer->connectInTensor(idx, randomSeedClone->id);
          }
        }
      }
    }
  }

  return randomSeedClone->id;
}

bool containsSeedTensor(std::set<TensorId> ids) {
  bool containsSeedFromHost =
      std::find(ids.begin(),
                ids.end(),
                GetRandomSeedOp::getStreamedSeedTensorId()) != ids.end();
  bool containsUpdatedSeed =
      std::find(ids.begin(),
                ids.end(),
                GetRandomSeedOp::getUpdatedSeedTensorId()) != ids.end();

  if (containsSeedFromHost || containsUpdatedSeed) {
    return true;
  }
  return false;
}

void adjustStashCandidatesForRandomSeed(
    Graph &graph,
    std::set<TensorId> &toStashCandidateTensors) {
  auto &ir = graph.getIr();
  if (RandomSetup::hasRandomSeed(ir) &&
      containsSeedTensor(toStashCandidateTensors)) {
    // Neither the input or the output of a GetRandomSeedOp should be stashed.
    auto getRandomSeedOp = findGetRandomSeedOp(graph);
    toStashCandidateTensors.erase(getRandomSeedOp->inId(0));
    toStashCandidateTensors.erase(getRandomSeedOp->outId(0));
    // Instead, we need to clone the output of the random seed op and stash
    // that.
    auto stashableRandomSeed = createStashableRandomSeed(getRandomSeedOp);
    toStashCandidateTensors.insert(stashableRandomSeed);
  }
}

IncrementModOp *insertCounterUpdate(Graph &graph,
                                    int stashSize,
                                    TensorId stashCounterInnerGraphId,
                                    TensorId updatedStashCounterInnerGraphId,
                                    const Op::Settings &settings) {
  return graph.createConnectedOp<IncrementModOp>(
      {{IncrementModOp::getInIndex(), stashCounterInnerGraphId}},
      {{IncrementModOp::getOutIndex(), updatedStashCounterInnerGraphId}},
      Onnx::CustomOperators::IncrementMod_1,
      static_cast<double>(1),
      static_cast<double>(stashSize),
      settings);
}

} // namespace

void Pipeline::checkOpsPipelineStage(Graph &graph) {
  // return the pipeline stage or -1 if not set
  // throw error if pipeline stage is negative
  auto getPipelineStage = [](auto x) -> PipelineStage {
    if (x->hasPipelineStage()) {
      auto ps = x->getPipelineStage();
      if (ps < 0) {
        throw error("Op has bad pipeline stage {}", ps);
      }
      return ps;
    } else {
      return -1;
    }
  };

  // collect all ops in each pipeline stage
  std::map<PipelineStage, std::vector<Op *>> pipelineStages;

  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (!op->isConvertibleTo<IpuCopyOp>()) {
      auto ps = getPipelineStage(op);
      pipelineStages[ps].push_back(op);
    }
  }

  // if no ops have had the pipeline stage attribute set, set it to the virtual
  // graph id
  if (pipelineStages.size() == 1 && pipelineStages.count(-1) != 0) {
    for (auto &id_op : graph.getOps()) {
      auto op = id_op.second.get();
      if (!op->isConvertibleTo<IpuCopyOp>()) {
        auto vgraphid = op->getVirtualGraphId();
        op->setPipelineStage(vgraphid);
      }
    }
  }

  std::ostringstream oss;
  oss << "For all IpuCopyOps (other than those copying optimizer tensors):\n"
      << " (1) set pipeline stage\n"
      << " (2) insert topological constraint that it precedes "
      << "all non-IpuCopyOps in each pipeline stage";
  logging::transform::debug(oss.str());

  for (auto &id_op : graph.getOps()) {
    if (auto copyOp = dynamic_cast<IpuCopyOp *>(id_op.second.get())) {
      // Copies of optimizer Tensors do not run in the main program fragment
      if (!copyOp->copiesOptimizerTensors()) {
        // (1) set pipeline stage
        setIPUCopyPipelineStage(copyOp);
        // (2) insert topological constraint
        std::vector<Op *> nonCopyOps =
            pipelineStages.at(copyOp->getPipelineStage());
        std::vector<Op *> mainProgramOps;
        for (auto op : nonCopyOps) {
          if (op->settings.executionContext == ExecutionContext::Normal) {
            mainProgramOps.push_back(op);
          }
        }
        graph.topoCons->insert({{
            copyOp,        // Key
            mainProgramOps // OpsBeforeKey
        }});
      }
    }
  }
}

bool Pipeline::checkIsFullRecompute(Graph &graph) {
  auto &ir = graph.getIr();
  return ir.canTrain() && (ir.getSessionOptions().autoRecomputation ==
                           RecomputationType::Pipeline);
}

void Pipeline::setFinalFwdStageRecomputation(Graph &graph) {
  // This annotation pass will try to set the Ops between
  // the topologically final Checkpoints and the loss
  // to NOT be recomputed. This avoids a program where
  // operations are run twice in a row with no benefit to
  // liveness.
  std::map<PipelineStage, std::pair<bool, bool>> prePostLoss;
  // Find PipelineStages with SchedulePreLoss Yes and No.
  // This should only be where the final loss is executed.
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->settings.executionContext == ExecutionContext::Normal) {
      auto pStage = op->getPipelineStage();
      prePostLoss[pStage].first |=
          op->scheduledPreLoss == ScheduledPreLoss::Yes;
      prePostLoss[pStage].second |=
          op->scheduledPreLoss == ScheduledPreLoss::No;
    }
  }

  nonstd::optional<PipelineStage> finalFwdStage;
  for (auto &pre_post : prePostLoss) {
    if (pre_post.second.first && pre_post.second.second) {
      if (finalFwdStage) {
        throw internal_error(
            "[Pipeline::setFinalFwdStageRecomputation] Found more than one "
            "PipelineStage with ScheduledPreLoss::Yes and "
            "ScheduledPreLoss::No Ops. Only the stage with the final loss "
            "should have this property.");
      }
      finalFwdStage = pre_post.first;
    }
  }

  if (finalFwdStage) {
    // Iterate through Ops in topological order.
    // Add to frontier when RecomputeType::Checkpoint && ScheduledPreLoss::Yes
    // is reached. If an Op in the frontier is in this new Op's history,
    // remove it from the frontier. Finally propagate
    // RecomputeType::Checkpoint from the frontier.
    PipelineStage pStage = (*finalFwdStage);
    logging::trace("[Pipeline::setFinalFwdStageRecomputation] Setting "
                   "FinalStage Recompute for {}",
                   pStage);

    std::map<OpId, std::set<OpId>> paths;
    std::vector<Op *> frontier;

    auto hasSamePipelineStage = [](Op *op, PipelineStage pStage) {
      if (op->isConvertibleTo<IpuCopyOp>()) {
        std::set<PipelineStage> allPStages;

        for (const Tensor *t : op->output->tensors()) {
          auto pStages = t->getPipelineStages();
          allPStages.insert(pStages.begin(), pStages.end());
        }
        return allPStages.find(pStage) != allPStages.end();
      } else {
        return op->getPipelineStage() == pStage;
      }
    };

    auto sameContextAndStage = [&pStage, &hasSamePipelineStage](Op *op) {
      return op->scheduledPreLoss == ScheduledPreLoss::Yes &&
             op->settings.executionContext == ExecutionContext::Normal &&
             hasSamePipelineStage(op, pStage);
    };

    auto pruneFromFrontier = [&frontier](const std::set<OpId> &path) {
      auto it = frontier.begin();
      while (it != frontier.end()) {
        if (path.find((*it)->id) != path.end()) {
          logging::trace("[Pipeline::setFinalFwdStageRecomputation] Pruned "
                         "from frontier {}",
                         (*it)->debugName());
          it = frontier.erase(it);
        } else {
          ++it;
        }
      }
    };

    auto addConsumerPaths = [&sameContextAndStage,
                             &paths](Op *op, const std::set<OpId> path) {
      for (auto t : op->output->tensors()) {
        for (auto con : t->consumers.getOps()) {
          if (sameContextAndStage(con)) {
            paths[con->id].insert(path.begin(), path.end());
          }
        }
      }
    };

    for (auto op : graph.getOpSchedule({}, RequireOptimalSchedule::No)) {
      if (sameContextAndStage(op)) {
        auto path = paths[op->id];
        path.insert(op->id);
        addConsumerPaths(op, path);
        if (op->settings.recomputeType == RecomputeType::Checkpoint) {
          pruneFromFrontier(path);
          frontier.push_back(op);
          logging::trace(
              "[Pipeline::setFinalFwdStageRecomputation] Added to frontier {}",
              op->debugName());
        }
      }
    }

    if (frontier.empty()) {
      // Since IpuCopyOps are checkpointed in Pipeline::setRecompute, we always
      // expect a non-empty frontier.
      throw internal_error("[Pipeline::setFinalFwdStageRecomputation] Frontier "
                           "is empty. No checkpoint operations have been found "
                           "on the final forward PipelineStage.");
    }

    if (logging::shouldLog(logging::Module::popart, logging::Level::Trace)) {
      logging::trace("[Pipeline::setFinalFwdStageRecomputation] Frontier:");
      for (auto op : frontier) {
        logging::trace("[Pipeline::setFinalFwdStageRecomputation]   {} {} {}",
                       op->id,
                       op->opid,
                       op->name());
      }
    }

    std::set<OpId> visited;
    for (auto op : frontier) {
      visited.insert(op->id);
    }

    while (!frontier.empty()) {
      auto op = frontier.back();
      frontier.pop_back();
      op->settings.recomputeType = RecomputeType::Checkpoint;
      for (auto t : op->output->tensors()) {
        for (auto con : t->consumers.getOps()) {
          if (visited.find(con->id) == visited.end() &&
              sameContextAndStage(con)) {
            frontier.push_back(con);
            visited.insert(con->id);
          }
        }
      }
    }
  } else {
    if (graph.getIr().isTraining()) {
      logging::warn("[Pipeline::setFinalFwdStageRecomputation] Could not find "
                    "final forward pipelineStage.");
    }
  }
}

bool Pipeline::inplaceRestoreRequiredForRecompute(Op *op) {
  if (dynamic_cast<RestoreInplaceOp *>(op)) {
    return dynamic_cast<RestoreInplaceOp *>(op)->requiredForRecompute;
  }
  return false;
}

bool Pipeline::inplaceRecomputationConflict(Op *op, InIndex in, OutIndex out) {
  // PipelineCycles are not represented explicity in the IR. As such certain
  // cases of inplace conflict must be handled here.
  // A pipeline cycle with recomputation can cause tensors to be overwritten.
  // On a single IPU a pipelineCycle will be lowered as:
  //    {stageA}, {stageA_recompute, stageB}, ipuCopies
  //
  // This is represented in the IR as:
  //    {StageA} ipuCopies {stageA_recompute, stageB}    (note: stageA_recompute
  //    might be implicit)
  //
  // When inplacing operations in the IR, tensors in stageA that are consumed by
  // ipuCopies will not present any conflicts as the tensors will already have
  // been consumed before stageA_recompute where they will be modified.

  // However, in the lowered version there is a conflict as {stageA_recompute}
  // is executed before ipuCopies. As such there is a constaint that any Tensor
  // produced by recomputation and is consumed by an ipuCopy must not be
  // aliased.
  //
  // The case where there is a direct producer->consumer relationship is handled
  // by insertClonesBeforeIpuCopyConsumers.

  auto modifiedByImplicitRecompute = [](Tensor *t) {
    return t->isImplicitRecomputeTensor();
  };
  auto isConsumedByIpuCopy = [](Tensor *t) {
    for (Op *consumer : t->consumers.getOps()) {
      if (consumer->isIpuCopyOp()) {
        return true;
      }
    }
    return false;
  };

  bool implicitRecomputeIn =
      op->input->tensor(in)->anyAlias(modifiedByImplicitRecompute);
  bool implicitRecomputeOut =
      op->output->tensor(out)->anyAlias(modifiedByImplicitRecompute);

  bool ipuCopyIn  = op->input->tensor(in)->anyAlias(isConsumedByIpuCopy);
  bool ipuCopyOut = op->output->tensor(out)->anyAlias(isConsumedByIpuCopy);

  return (implicitRecomputeIn && ipuCopyOut) ||
         (implicitRecomputeOut && ipuCopyIn);
}

bool Pipeline::apply(Graph &graph) const {
  auto &ir = graph.getIr();

  // First, some checks that pipelining is compatible with other user options:

  // 1. Pipelining uses the virtual graph API. This must be enabled
  if (!ir.virtualGraphsEnabled()) {
    throw error("Pipelining requires the 'virtualGraphMode' session option "
                "to not be VirtualGraphMode::Off.");
  }

  checkOpsPipelineStage(graph);

  // 2. Currently user-annotated recomputation is not supported with pipelining
  // (TODO T9575)
  if (ir.getSessionOptions().implicitPipeliningEnabled() &&
      graph.hasUserRecomputeOps()) {
    throw error("When pipelining is enabled, user annotation for recomputation "
                "is not allowed");
  }

  // 3. Forward layers must be sharded with increasing IPU index
  //    Examples violating this:
  //      Consider the fwd Graph : Op0 -> Op1 -> Op2 -> Op3
  //          e.g. 1) IPU0 : {Op2, Op3}, IPU1 : {Op0, Op1}
  //          e.g. 2) IPU0 : {Op0, Op2}, IPU1 : {Op1, Op3}

  // The checks:
  // 3.2 Copies in the correct direction

  chainCopiesTransform(graph);

  // Other sharding assumptions to check:

  // 4. Ir stream tensors cannot be consumed by ops on multiple IPUs
  // (not relevant for explicit pipelining)
  if (ir.getSessionOptions().implicitPipeliningEnabled()) {
    for (TensorId tid : graph.getTensors().getIds(TensorType::Stream)) {
      auto tensor = graph.getTensors().get(tid);
      std::set<VGraphId> virtualGraphs;
      for (auto c : tensor->consumers.getOps()) {
        virtualGraphs.insert(getVirtualGraphIdOrSourceIpu(c, tensor));
      }

      if (virtualGraphs.size() > 1) {
        throw error("For pipelining, stream tensors can only be streamed "
                    "directly onto a single IPU");
      }
    }
  }

  // Merge Consecutive stages.
  mergeConsecutivePipelineStages(graph);

  // 5. There must be enough mini-batches of data to fill the pipeline
  int64_t numPipelineStages = ir.getNumPipelineStages();
  if (ir.getSessionOptions().enableGradientAccumulation) {
    if (ir.getSessionOptions().accumulationFactor < numPipelineStages) {
      // For replicated graphs we are replicating the entire pipeline, so these
      // conditions still hold.
      throw error("For pipelining, depth (gradient accumulation factor) must "
                  "equal at least the number of pipeline stages ({})",
                  numPipelineStages);
    }
  } else {
    int64_t bps = static_cast<int64_t>(ir.getDataFlow().batchesPerStep());
    if (bps < numPipelineStages) {
      throw error("For pipelining, depth (batchesPerStep) must equal at least "
                  "the number of pipeline stages ({})",
                  numPipelineStages);
    }
  }

  // 6. Contiguate the IPUCopies
  contiguateIpuCopies(graph);

  if (ir.getSessionOptions().explicitPipeliningEnabled()) {
    // Get the inner loop subgraph that is to be pipelined

    // 7. Add dynamic stash and restore ops
    addDynamicStashAndRestoreOps(graph);

    // 8. Decompose loop to enable explicit pipelining
    return applyExplicit(graph);
  } else {
    // 7. Add implicit stash and restore ops
    return addStashRestoreOps(graph);
  }
}

bool Pipeline::contiguateIpuCopies(Graph &graph) const {
  auto getIpuCopyOps = [&graph] {
    // contiguating IpuCopyOps
    std::vector<popart::IpuCopyOp *> ipuCopies;
    for (auto &op_pair : graph.getOps()) {
      auto ipuCopyOp = dynamic_cast<popart::IpuCopyOp *>(op_pair.second.get());
      if (ipuCopyOp &&
          (ipuCopyOp->settings.executionContext == ExecutionContext::Normal ||
           ipuCopyOp->settings.executionContext ==
               ExecutionContext::Subgraph)) {
        ipuCopies.push_back(ipuCopyOp);
      }
    }
    return ipuCopies;
  };

  ContiguateIpuCopyIndicesPattern contiguator;
  for (auto ipuCopyOp : getIpuCopyOps()) {
    if (!ipuCopyOp->isExcludedFromPattern(&contiguator) &&
        contiguator.matches(ipuCopyOp)) {
      logging::transform::debug("Contiguating {}", ipuCopyOp->debugName());
      contiguator.apply(ipuCopyOp);
    }
  }
  graph.getIr().updateVertices();

  // verify that all IpuCopies are contiguous
  for (auto ipuCopyOp : getIpuCopyOps()) {
    if (!ipuCopyOp->copiesOptimizerTensors()) {
      auto sourceIpu = ipuCopyOp->getPipelineStage();
      auto destIpu =
          *ipuCopyOp->outTensor(0)->consumers.findLowestPipelineStage();
      auto delta = destIpu - sourceIpu;
      // only copies of optimizer may be non contiguous
      if (delta != 1 && delta != -1) {
        std::stringstream ss;
        ss << logging::format(
            "IpuCopy {} is not contiguous. It copies from IPU {} to "
            "IPU {}. Failed to contiguate all IpuCopyOps",
            ipuCopyOp->debugName(),
            sourceIpu,
            destIpu);
        ss << logging::format("\nin tensor 0: {}",
                              ipuCopyOp->inTensor(0)->str());
        ss << logging::format("\nin tensor 0 producer pipeline stage: {}",
                              sourceIpu);
        ss << logging::format("\nout tensor 0: {}",
                              ipuCopyOp->outTensor(0)->str());
        ss << logging::format(
            "\nout tensor 0 lowest consumer pipeline stage: {}", destIpu);
        throw internal_error(ss.str());
      }
    }
  }

  return true;
}

bool Pipeline::applyExplicit(Graph &innerLoopSubgraph) const {
  ExplicitPipelineHelper explicitPipeline(innerLoopSubgraph);
  explicitPipeline.createExplicitPipeline();
  return true;
}

PipelineStashInfo
Pipeline::prepareForStashing(Graph &graph,
                             std::set<TensorId> toStashCandidateTensors) const {
  auto &ir             = graph.getIr();
  bool isFullRecompute = Pipeline::checkIsFullRecompute(graph);
  bool isExplicit =
      graph.getIr().getSessionOptions().explicitPipeliningEnabled();

  std::set<TensorId> toStashTensors;
  // StashTensorId -> std::pair<StashRefOp, RestoreRefOps>
  std::map<TensorId, std::pair<Op *, std::vector<Op *>>> stashRestoreRefOps;
  // If there is no recomputation, then the candidates for stashing will all be
  // stashed.
  if (!ir.autoRecomputationEnabled()) {
    toStashTensors = toStashCandidateTensors;
  }

  else if (isExplicit) {
    // Explicit pipelining uses explicit recomputation, so by this point in
    // time, all the recomputation ops have already been inserted into the IR.
    // Therefore we can filter recomputation based on the recomputed ops.
    toStashTensors = toStashCandidateTensors;
  } else {
    // If there is recomputation, the candidate set is reduced.
    //
    // Candidate Tensors which can be recomputed from other stashing candidates,
    // are filtered out, and their producers are set to Recompute.
    //
    // The only exceptions are candidate stashing Tensors which are copied to
    // another IPU : these must be stashed even if they recomputable. This
    // guarantees that the correct Tensor is copied after fwd and bwd have
    // executed.
    //
    // Algorithm : initialize all pre-loss Ops to be Recompute, and then set to
    // Checkpoint if (1) cannot be computed from previous Stashed Tensors or (2)
    // must be copied to next IPU.
    setRecomputation(graph, toStashCandidateTensors);

    std::set<TensorId> checkpointTensors;

    logging::transform::debug(
        "Reducing the set of stashing candidate Tensors for recomputation");

    // Filter stash candidates: only stash Checkpoint Ops
    for (auto tid : toStashCandidateTensors) {
      auto tensor = graph.getTensors().get(tid);
      if (!tensor->hasProducer() ||
          tensor->getProducer()->settings.recomputeType !=
              RecomputeType::Recompute) {
        // For isFullRecompute if a stash candidate doesn't have a
        // restoreReference then it is not required for recomputation during the
        // backwards pass.
        if (isFullRecompute) {
          auto stashRef = getStashReferenceOp(tensor);
          auto restoreRefs =
              findDescendentsOnDifferentPipelineStages(tensor, stashRef);
          if (restoreRefs.size() == 0) {
            logging::transform::debug("Discarding Stash candidate tensor {} as "
                                      "no restore reference found.",
                                      tid);
            if (tensor->getPipelineStages().size() == 1) {
              // Tensors that only have one PipelineStage should not be stashed
              // however we should still keep track of them as Checkpoint
              // tensors. This allows for "standard" recompute to be possible on
              // the final forward pipelineStage.
              checkpointTensors.insert(tid);
            }
            continue;
          } else if (restoreRefs.size() > 1) {
            size_t nRecomputeRequiredRestores = 0;
            auto consumerOps                  = tensor->consumers.getOps();
            for (auto restoreRef : restoreRefs) {
              if (std::find(consumerOps.begin(),
                            consumerOps.end(),
                            restoreRef) == consumerOps.end()) {
                nRecomputeRequiredRestores++;
              }
            }
            // TODO T30014 : Requires extending recompute such that a recomp
            //               fragment can run once per PipelineStage (instead of
            //               only once)
            if (nRecomputeRequiredRestores > 1) {
              throw error(
                  "SessionOptions::autoRecomputation == "
                  "RecomputationType::Pipeline is not compatible with the "
                  "partitioning of operations over pipeline stages. To-stash "
                  "Tensor '{}' must be restored for recomputation of a "
                  "descendent that is not a direct consumer on more than 1 "
                  "PipelineStage, but this is currently not supported",
                  tid);
            } else {
              stashRestoreRefOps.insert({tid, {stashRef, restoreRefs}});
            }
          } else {
            stashRestoreRefOps.insert({tid, {stashRef, restoreRefs}});
          }
        }
        toStashTensors.insert(tid);
      }
    }

    // If the set of stash candidates has been reduced, recomputation needs to
    // be reset.
    checkpointTensors.insert(toStashTensors.begin(), toStashTensors.end());
    if (checkpointTensors.size() != toStashCandidateTensors.size()) {
      setRecomputation(graph, checkpointTensors);
    }
  }

  logging::transform::debug(
      "[Pipeline::prepareForStashing] Final stash tensors: {}", toStashTensors);

  PipelineStashInfo info;
  info.toStashTensors     = toStashTensors;
  info.stashRestoreRefOps = stashRestoreRefOps;

  return info;
}

TensorId Pipeline::addDynamicStashOpForTensor(
    Graph &graph,
    const PipelineDynamicStashInfo &info) const {
  auto &ir                         = graph.getIr();
  LoopOp *innerLoopOp              = MainLoops::getInnerLoopOp(ir);
  auto &pipelineMainLoopOuterGraph = innerLoopOp->getGraph();

  Op::Settings outerSettings(pipelineMainLoopOuterGraph, "");
  Op::Settings stashSettings(graph, "");

  // Info for the stash counters
  TensorInfo counterInfo(DataType::INT32, {});

  // Add the stash size as the leading dimension of the stash tensor
  TensorInfo stashTensorInfo = info.tensor->info;
  Shape tensorShape          = info.tensor->info.shape();
  Shape stashTensorShape;
  stashTensorShape.reserve(stashTensorInfo.shape().size() + 1);
  stashTensorShape.push_back(info.stashSize);
  stashTensorShape.insert(
      stashTensorShape.end(), tensorShape.begin(), tensorShape.end());
  stashTensorInfo.set(info.tensor->info.dataType(), stashTensorShape);

  auto stashTensorInnerGraphId = addScope(graph, info.stashTensorBaseId);
  auto stashTensorOuterGraphId =
      addScope(pipelineMainLoopOuterGraph, info.stashTensorBaseId);
  auto updatedStashTensorInnerGraphId =
      ir.createIntermediateTensorId(stashTensorInnerGraphId);
  auto updatedStashTensorOuterGraphId =
      ir.createIntermediateTensorId(stashTensorOuterGraphId);

  logging::transform::trace(
      "[Pipeline::addDynamicStashOpForTensor] Creating stash "
      "tensorIds: {} -> [{} -> {}] -> {}",
      stashTensorOuterGraphId,
      stashTensorInnerGraphId,
      updatedStashTensorInnerGraphId,
      updatedStashTensorOuterGraphId);

  auto stashCounterInnerGraphId = addScope(graph, info.stashCounterBaseId);
  auto stashCounterOuterGraphId =
      addScope(pipelineMainLoopOuterGraph, info.stashCounterBaseId);
  auto updatedStashCounterInnerGraphId =
      ir.createIntermediateTensorId(stashCounterInnerGraphId);
  auto updatedStashCounterOuterGraphId =
      ir.createIntermediateTensorId(stashCounterOuterGraphId);

  logging::transform::trace(
      "[Pipeline::addDynamicStashOpForTensor] Creating stash counter "
      "tensorIds: {} -> [{} -> {}] -> {}",
      stashCounterOuterGraphId,
      stashCounterInnerGraphId,
      updatedStashCounterInnerGraphId,
      updatedStashCounterOuterGraphId);

  // Create the stash counter in the outer graph (once per stash)
  auto stashCounterInitOp =
      pipelineMainLoopOuterGraph.createConnectedOp<InitOp>(
          {},
          {{InitOp::getOutIndex(), stashCounterOuterGraphId}},
          Onnx::CustomOperators::Init_1,
          counterInfo,
          TensorType::ActGrad,
          InitType::Zero,
          outerSettings.copy("Init_" + info.stashCounterBaseId));

  // Create the stash in the outer graph
  auto stashTensorInitOp = pipelineMainLoopOuterGraph.createConnectedOp<InitOp>(
      {},
      {{InitOp::getOutIndex(), stashTensorOuterGraphId}},
      Onnx::CustomOperators::Init_1,
      stashTensorInfo,
      TensorType::ActGrad,
      InitType::NoInit,
      outerSettings.copy("Init_" + info.stashTensorBaseId));

  // Loop input stash counter
  innerLoopOp->addLoopInput(LoopOp::getFirstInputInIndex(),
                            stashCounterOuterGraphId,
                            stashCounterInnerGraphId,
                            false);

  // Loop input stash tensor (implicit, modified input)
  auto stashInIndex = innerLoopOp->input->n();
  innerLoopOp->addLoopInput(
      stashInIndex, stashTensorOuterGraphId, stashTensorInnerGraphId, false);
  innerLoopOp->addModified(
      stashInIndex,
      {view::Region::getFull(
          innerLoopOp->input->tensor(stashInIndex)->info.shape(),
          view::AccessType::ReadWrite)});

  // Dynamic update the stash inplace
  auto dynamicUpdateOp = graph.createConnectedOp<DynamicUpdateInplaceOp>(
      {{DynamicUpdateInplaceOp::getUpdateInIndex(), stashTensorInnerGraphId},
       {DynamicUpdateInplaceOp::getIndexInIndex(), stashCounterInnerGraphId},
       {DynamicUpdateInplaceOp::getInIndex(), info.tensor->id}},
      {{DynamicUpdateInplaceOp::getOutIndex(), updatedStashTensorInnerGraphId}},
      Onnx::CustomOperators::DynamicUpdateInplace,
      std::vector<int64_t>{0},
      std::vector<int64_t>{1},
      false,
      stashSettings.copy("DynamicUpdate_" + info.stashTensorBaseId));

  // Update the stash counter
  Op *stashCounterUpdateOp = insertCounterUpdate(
      graph,
      info.stashSize,
      stashCounterInnerGraphId,
      updatedStashCounterInnerGraphId,
      stashSettings.copy("Counter_" + info.stashTensorBaseId));

  // Loop carry the updated counter
  innerLoopOp->addLoopOutput(LoopOp::getFirstOutputOutIndex(),
                             updatedStashCounterOuterGraphId,
                             updatedStashCounterInnerGraphId,
                             false);

  stashCounterInitOp->setVirtualGraphId(
      getVirtualGraphIdOrSourceIpu(info.stashRefOp, info.tensor));
  stashTensorInitOp->setVirtualGraphId(
      getVirtualGraphIdOrSourceIpu(info.stashRefOp, info.tensor));

  stashCounterUpdateOp->setVirtualGraphId(
      getVirtualGraphIdOrSourceIpu(info.stashRefOp, info.tensor));
  stashCounterUpdateOp->setPipelineStage(info.stashRefOp->getPipelineStage());

  dynamicUpdateOp->setVirtualGraphId(
      getVirtualGraphIdOrSourceIpu(info.stashRefOp, info.tensor));
  dynamicUpdateOp->setPipelineStage(info.stashRefOp->getPipelineStage());

  return stashTensorInnerGraphId;
}

TensorId
Pipeline::addDynamicRestoreOpForTensor(Graph &graph,
                                       const PipelineDynamicStashInfo &info,
                                       size_t restoreRefOpIndex,
                                       TensorId stashTensorInnerGraphId,
                                       TensorId lastRestoreTensorId) const {

  auto &ir                         = graph.getIr();
  LoopOp *innerLoopOp              = MainLoops::getInnerLoopOp(ir);
  auto &pipelineMainLoopOuterGraph = innerLoopOp->getGraph();

  Op::Settings outerSettings(pipelineMainLoopOuterGraph, "");
  Op::Settings restoreSettings(graph, "");

  Op *restoreRefOp = info.restoreRefOps.at(restoreRefOpIndex);

  // Info for the stash and restore counters
  TensorInfo counterInfo(DataType::INT32, {});

  TensorId restoreCounterInnerGraphId =
      ir.createIntermediateTensorId(addScope(graph, info.stashCounterBaseId));
  TensorId restoreCounterOuterGraphId = ir.createIntermediateTensorId(
      addScope(pipelineMainLoopOuterGraph, info.stashCounterBaseId));
  TensorId updatedRestoreCounterInnerGraphId =
      ir.createIntermediateTensorId(addScope(graph, info.stashCounterBaseId));
  TensorId updatedRestoreCounterOuterGraphId = ir.createIntermediateTensorId(
      addScope(pipelineMainLoopOuterGraph, info.stashCounterBaseId));

  logging::transform::trace("[Pipeline::addDynamicRestoreOpForTensor] Creating "
                            "tensorIds: {} -> [{} -> {}] -> {}",
                            restoreCounterOuterGraphId,
                            restoreCounterInnerGraphId,
                            updatedRestoreCounterInnerGraphId,
                            updatedRestoreCounterOuterGraphId);

  // Create the restore counter in the outer graph (once per restore)
  auto restoreCounterInitOp =
      pipelineMainLoopOuterGraph.createConnectedOp<InitOp>(
          {},
          {{InitOp::getOutIndex(), restoreCounterOuterGraphId}},
          Onnx::CustomOperators::Init_1,
          counterInfo,
          TensorType::ActGrad,
          InitType::Zero,
          outerSettings.copy("Init_" + info.stashCounterBaseId + "_" +
                             std::to_string(restoreRefOpIndex)));

  restoreCounterInitOp->setVirtualGraphId(
      getVirtualGraphIdOrSourceIpu(restoreRefOp, info.tensor));

  // Loop input restore counter
  innerLoopOp->addLoopInput(LoopOp::getFirstInputInIndex(),
                            restoreCounterOuterGraphId,
                            restoreCounterInnerGraphId,
                            false);

  // Update the restore counter
  Op *restoreCounterUpdateOp = insertCounterUpdate(
      graph,
      info.stashSize,
      restoreCounterInnerGraphId,
      updatedRestoreCounterInnerGraphId,
      restoreSettings.copy("Counter_" + info.stashTensorBaseId + "_" +
                           std::to_string(restoreRefOpIndex)));

  restoreCounterUpdateOp->setVirtualGraphId(
      getVirtualGraphIdOrSourceIpu(restoreRefOp, info.tensor));
  restoreCounterUpdateOp->setPipelineStage(restoreRefOp->getPipelineStage());

  // Loop carry the updated counter
  innerLoopOp->addLoopOutput(LoopOp::getFirstOutputOutIndex(),
                             updatedRestoreCounterOuterGraphId,
                             updatedRestoreCounterInnerGraphId,
                             false);

  TensorId toRestoreTensorId =
      lastRestoreTensorId.empty() ? info.tensor->id : lastRestoreTensorId;
  TensorId restoredTensorId = ir.createIntermediateTensorId(info.tensor->id);

  logging::transform::trace(
      "[Pipeline::addDynamicRestoreOpForTensor] Restoring tensor "
      "{}: {} -> {}",
      info.tensor->id,
      toRestoreTensorId,
      restoredTensorId);

  DynamicSliceOp *dynamicSliceOp = graph.createConnectedOp<DynamicSliceOp>(
      {{DynamicSliceOp::getInIndex(), stashTensorInnerGraphId},
       {DynamicSliceOp::getIndexInIndex(), restoreCounterInnerGraphId},
       {DynamicSliceOp::getSliceInIndex(), toRestoreTensorId}},
      {{DynamicSliceOp::getOutIndex(), restoredTensorId}},
      Onnx::CustomOperators::DynamicSlice_1,
      std::vector<int64_t>{0},
      std::vector<int64_t>{1},
      false,
      restoreSettings.copy("DynamicSlice_" + info.stashTensorBaseId + "_" +
                           std::to_string(restoreRefOpIndex)));

  dynamicSliceOp->setVirtualGraphId(
      getVirtualGraphIdOrSourceIpu(restoreRefOp, info.tensor));
  dynamicSliceOp->setPipelineStage(restoreRefOp->getPipelineStage());

  // Disconnect tensor from all consumers in the restore op's pipeline stage,
  // reconnect to restoreId

  for (Op *consumer : info.consumers) {
    if (consumer->getPipelineStage() == dynamicSliceOp->getPipelineStage()) {
      for (auto inIndex : consumer->input->indicesMap().at(info.tensor)) {
        consumer->disconnectInTensor(inIndex, info.tensor);
        consumer->connectInTensor(inIndex, restoredTensorId);
      }
    } else {
      logging::transform::trace("[Pipeline::addDynamicRestoreOpForTensor] Not "
                                "connecting consumer {} to DynamicSliceOp {} "
                                "as they have different pipeline stages.",
                                consumer->debugName(),
                                dynamicSliceOp->debugName());
    }
  }

  return restoredTensorId;
}

void Pipeline::addDynamicRestoreOpsForTensor(
    Graph &graph,
    const PipelineDynamicStashInfo &info,
    TensorId stashTensorInnerGraphId) const {

  TensorId lastRestoreTensorId = "";
  for (size_t restoreRefOpIndex = 0;
       restoreRefOpIndex < info.restoreRefOps.size();
       restoreRefOpIndex++) {
    lastRestoreTensorId = addDynamicRestoreOpForTensor(graph,
                                                       info,
                                                       restoreRefOpIndex,
                                                       stashTensorInnerGraphId,
                                                       lastRestoreTensorId);
  }
}

void Pipeline::addDynamicStashAndRestoreOpsForTensor(
    Graph &graph,
    const PipelineStashInfo &pipelineStashInfo,
    TensorId tid) const {
  auto &ir = graph.getIr();

  PipelineDynamicStashInfo info;

  info.tensor = graph.getTensors().get(tid);
  // Important to capture the consumers before inserting new operations
  info.consumers = info.tensor->consumers.getOps();

  if (info.tensor->consumers.getOps().empty()) {
    throw internal_error("[Pipeline::addDynamicStashOpsForTensor] Request to "
                         "stash tensor {} with no "
                         "consumers, bailing",
                         info.tensor->str());
  }

  if (pipelineStashInfo.stashRestoreRefOps.find(tid) !=
      pipelineStashInfo.stashRestoreRefOps.end()) {
    auto refs          = pipelineStashInfo.stashRestoreRefOps.at(tid);
    info.stashRefOp    = refs.first;
    info.restoreRefOps = refs.second;
  } else {
    info.stashRefOp    = getStashReferenceOp(info.tensor);
    info.restoreRefOps = getRestoreReferenceOps(info.tensor, info.stashRefOp);
  }

  for (Op *restoreRefOp : info.restoreRefOps) {
    info.restoreStages.push_back(restoreRefOp->getPipelineStage());
  }

  // The largest PipelineStage of the restoreRefOps determines stash size
  info.stashSize =
      *std::max_element(info.restoreStages.begin(), info.restoreStages.end()) -
      info.stashRefOp->getPipelineStage() + 1;

  logging::transform::debug(
      "[Pipeline::addDynamicStashOpsForTensor] Adding stash of "
      "size {} of activations {} for "
      "pipelining. Stash stage: {}, restore stages: {}",
      info.stashSize,
      info.tensor->id,
      info.stashRefOp->getPipelineStage(),
      info.restoreStages);

  // Base TensorId for the stash tensor
  info.stashTensorBaseId = addPrefix(
      ir, removeScope(graph, info.tensor->id), reservedStashedPrefix());

  // Base TensorIds for the stash counter
  info.stashCounterBaseId = addPrefix(
      ir, removeScope(graph, info.tensor->id), reservedCounterPrefix());

  TensorId stashTensorInnerGraphId = addDynamicStashOpForTensor(graph, info);
  addDynamicRestoreOpsForTensor(graph, info, stashTensorInnerGraphId);
}

bool Pipeline::addDynamicStashAndRestoreOps(Graph &graph) const {
  auto &ir = graph.getIr();

  // Get graph which pipelineMainLoop is part of
  LoopOp *innerLoopOp = MainLoops::getInnerLoopOp(ir);

  logging::transform::debug(
      "[Pipeline::addDynamicStashOps] Adding pipeline stashes to {}",
      innerLoopOp->debugName());

  auto toStashCandidateTensors = getStashCandidateTensors(graph);
  adjustStashCandidatesForRandomSeed(graph, toStashCandidateTensors);
  auto pipelineStashInfo = prepareForStashing(graph, toStashCandidateTensors);

  if (!ir.getSessionOptions().explicitRecomputation) {
    throw error("[Pipeline::addDynamicStashOps] Explicit pipelining requires "
                "explicit recomputation to be enabled.");
  }

  for (auto &tid : pipelineStashInfo.toStashTensors) {
    addDynamicStashAndRestoreOpsForTensor(graph, pipelineStashInfo, tid);
  }

  innerLoopOp->setup();

  return true;
}

bool Pipeline::addStashRestoreOps(Graph &graph) const {
  auto &ir             = graph.getIr();
  bool isFullRecompute = Pipeline::checkIsFullRecompute(graph);

  auto toStashCandidateTensors = getStashCandidateTensors(graph);
  adjustStashCandidatesForRandomSeed(graph, toStashCandidateTensors);
  auto pipelineStashInfo = prepareForStashing(graph, toStashCandidateTensors);

  // For each Tensor to be stashed, create a single stash
  // and one or more (possible in-place) restore ops
  Op::Settings settings(graph, "");

  for (auto &tid : pipelineStashInfo.toStashTensors) {
    auto tensor = graph.getTensors().get(tid);

    if (tensor->consumers.getOps().empty()) {
      throw internal_error("request to stash Tensor {} with no "
                           "consumers, bailing",
                           tensor->str());
    }

    Op *stashRefOp;
    std::vector<Op *> restoreRefOps;
    if (pipelineStashInfo.stashRestoreRefOps.find(tid) !=
        pipelineStashInfo.stashRestoreRefOps.end()) {
      auto refs     = pipelineStashInfo.stashRestoreRefOps.at(tid);
      stashRefOp    = refs.first;
      restoreRefOps = refs.second;
    } else {
      stashRefOp    = getStashReferenceOp(tensor);
      restoreRefOps = getRestoreReferenceOps(tensor, stashRefOp);
    }

    std::vector<PipelineStage> restoreStages;
    for (Op *restoreRefOp : restoreRefOps) {
      restoreStages.push_back(restoreRefOp->getPipelineStage());
    }

    // The largest PipelineStage of the restoreRefOps determines stash size
    auto stashSize =
        *std::max_element(restoreStages.begin(), restoreStages.end()) -
        stashRefOp->getPipelineStage() + 1;

    logging::transform::debug("Adding stash of size {} of activations {} for "
                              "pipelining. Stash stage: {}, Restore stages {}",
                              stashSize,
                              tensor->id,
                              stashRefOp->getPipelineStage(),
                              restoreStages);

    // Stash
    auto stashOp_up = std::make_unique<StashOp>(
        Onnx::CustomOperators::Stash, stashSize, settings);
    auto stashOp = stashOp_up.get();
    graph.moveIntoGraph(std::move(stashOp_up));
    stashOp->setVirtualGraphId(
        getVirtualGraphIdOrSourceIpu(stashRefOp, tensor));
    stashOp->setPipelineStage(stashRefOp->getPipelineStage());
    stashOp->connectInTensor(StashOp::getInIndex(), tid);
    auto stashId = stashOp->getStashedTensorId();
    stashOp->createAndConnectOutTensor(StashOp::getOutIndex(), stashId);
    stashOp->setup();

    // Full Recomputation
    // If a preLoss stash tensors is consumed by an IpuCopy
    // it must not be inplace, but stashes needed for recomputation must be
    // inplace. To resolve this contradiction an IdentityOp is inserted between
    // the the stashed tensor and the IpuCopy
    if (isFullRecompute) {
      insertClonesBeforeIpuCopyConsumers(
          graph,
          tensor,
          getVirtualGraphIdOrSourceIpu(stashRefOp, tensor),
          stashRefOp->getPipelineStage());
    }

    auto tidConsumers = tensor->consumers.getOps();

    // Generate a map of consumers on each PipelineStage before
    // restore op(s) have been inserted. This will be used later to
    // insert topological constraints
    std::map<PipelineStage, std::vector<Op *>> preRestoreConsumers;
    for (auto tidConsumer : tidConsumers) {
      if (tidConsumer->hasPipelineStage()) {
        PipelineStage ps = tidConsumer->getPipelineStage();
        const auto found = preRestoreConsumers.find(ps);
        if (found == preRestoreConsumers.cend()) {
          preRestoreConsumers.emplace(ps, std::vector<Op *>{tidConsumer});
        } else {
          found->second.push_back(tidConsumer);
        }
      }
    }

    for (auto tidConsumer : tidConsumers) {
      // StashOp should be before all other consumers
      // (required for recompute to work)
      if (tidConsumer != stashOp) {
        graph.topoCons->insert(stashOp, tidConsumer);
      }
    }

    // Should op(s) be Restore (outplace) or RestoreInplace?
    bool isInplace = true;
    if (ir.isAnchored(tid)) {
      isInplace = false;
    } else {
      for (Op *tidConsumer : tidConsumers) {
        if (tidConsumer->isIpuCopyOp()) {
          isInplace = false;
        }
      }
    }

    // Recompute ops must be inplace, confirm:
    for (Op *tidConsumer : tidConsumers) {
      if (tidConsumer->settings.recomputeType == RecomputeType::Recompute) {
        if (isInplace == false) {
          throw error("A recompute Op consumes a stashed Tensor, {}. Therefore "
                      "the restoring of the tensor must be in-place. But some "
                      "previous logic has set the stashing to be non-inplace. "
                      "To resolve, either do not anchor this tensor (if "
                      "anchored), or change the 'autoRecomputation' "
                      "SessionOption from 'RecomputationType::Pipeline'",
                      tid);
        }
      }
    }

    for (size_t i = 0; i < restoreRefOps.size(); i++) {
      Op *restoreRefOp = restoreRefOps.at(i);
      RestoreOp *restoreOp;
      if (isInplace) {
        restoreOp = addNewRestoreInplaceOp(graph, stashSize);
        // RestoreInplaceOp has an extra input - the act tensor it is in-place
        // restoring.
        restoreOp->connectInTensor(RestoreInplaceOp::getActToRestoreInIndex(),
                                   tid);
      } else {
        restoreOp = addNewRestoreOp(graph, stashSize);
      }

      restoreOp->setVirtualGraphId(
          getVirtualGraphIdOrSourceIpu(restoreRefOp, tensor));
      restoreOp->setPipelineStage(restoreRefOp->getPipelineStage());
      restoreOp->connectInTensor(RestoreOp::getStashInIndex(), stashId);
      auto restoreId = restoreOp->getRestoredTensorId();
      if (i > 0) {
        // Uniquify restored TensorId
        restoreId += "_" + std::to_string(i);
      }
      restoreOp->createAndConnectOutTensor(RestoreOp::getRestoredActOutIndex(),
                                           restoreId);

      // Disconnect tid from all consumers in the restore op's pipeline stage,
      // reconnect to restoreId
      for (Op *tidConsumer : tidConsumers) {
        if (tidConsumer->getPipelineStage() == restoreOp->getPipelineStage()) {
          for (auto i : tidConsumer->input->indicesMap().at(tensor)) {
            tidConsumer->disconnectInTensor(i, tensor);
            tidConsumer->connectInTensor(i, restoreId);
          }
        } else {
          logging::transform::debug(
              "Not connecting consumer {} to restore op {} "
              "as they have different pipeline stages ",
              tidConsumer->str(),
              restoreOp->str());
        }
        if ((dynamic_cast<RestoreInplaceOp *>(restoreOp) &&
             tidConsumer->settings.recomputeType == RecomputeType::Recompute &&
             tidConsumer->getPipelineStage() < restoreOp->getPipelineStage())) {
          dynamic_cast<RestoreInplaceOp *>(restoreOp)->requiredForRecompute =
              true;
        }
      }

      // This InplaceRestoreOp may have no consumers as they are all implicit
      // due to recomputation. Therefore it must not be pruned.
      if (inplaceRestoreRequiredForRecompute(restoreOp)) {
        restoreOp->pruneable = false;
      }

      restoreOp->setup();

      bool noConsumersOfRestore =
          restoreOp->output->tensor(0)->consumers.getOps().empty();

      if (noConsumersOfRestore &&
          !inplaceRestoreRequiredForRecompute(restoreOp)) {
        // refresh consumers of tensor
        tidConsumers = tensor->consumers.getOps();

        bool noRecomputeConsumersOfStash = std::none_of(
            tidConsumers.cbegin(), tidConsumers.cend(), [](const Op *op) {
              return op->settings.recomputeType == RecomputeType::Recompute;
            });
        std::ostringstream oss3;
        oss3 << "The RestoreOp " << restoreOp->str() << " on pipeline stage "
             << restoreOp->getPipelineStage()
             << " has no consumers. This seems strange, so bailing. "
             << "noRecomputeConsumersOfStash = " << noRecomputeConsumersOfStash
             << " where the tensor being stashed is " << tensor->str();
        throw internal_error(oss3.str());
      }

      // Restore comes after Stash
      graph.topoCons->insert(stashOp, restoreOp);

      if (inplaceRestoreRequiredForRecompute(restoreOp)) {
        logging::debug("Inserting topocons for inplaceRestoreOp required for "
                       "implicit recomputation");
        restoreOp->settings.schedulePriority =
            std::numeric_limits<double>::lowest();
        for (auto op : findImplicitRecomputeDependants(restoreOp)) {
          if (restoreOp->id != op->id) {
            graph.topoCons->insert(restoreOp, op);
          }
        }
      } else {
        for (auto ps_tidConsumers : preRestoreConsumers) {
          PipelineStage ps               = ps_tidConsumers.first;
          std::vector<Op *> tidConsumers = ps_tidConsumers.second;

          // RestoreOp should be after all other consumers on a
          // lower PipelineStage (required for inplacing to work)
          if (ps < restoreOp->getPipelineStage()) {
            for (Op *tidConsumer : tidConsumers) {
              graph.topoCons->insert(tidConsumer, restoreOp);
            }
          }
        }
      }
    }
  }

  // Any tensor that is created by a recomputed op may be overwritten
  // by the recompute phase before it is copied to the next IPU, or back
  // to host (in the case of anchor tensors). So insert a identity
  // (clone) between the op and the copy.
  if (isFullRecompute) {
    for (auto &tid : graph.getTensors().getAllTensorIds()) {
      auto tensor = graph.getTensors().get(tid);

      // Stash tensors have already been covered above
      if (std::find(pipelineStashInfo.toStashTensors.cbegin(),
                    pipelineStashInfo.toStashTensors.cend(),
                    tid) != pipelineStashInfo.toStashTensors.cend()) {
        continue;
      }

      if (tensor->hasProducer()) {
        Op *producer = tensor->getProducer();
        if (producer->settings.recomputeType == RecomputeType::Recompute) {
          insertClonesBeforeIpuCopyConsumers(
              graph,
              tensor,
              getVirtualGraphIdOrSourceIpu(producer, tensor),
              producer->getPipelineStage());

          if (ir.isAnchored(tid) && tensor->consumers.getTotal() > 0) {
            insertCloneBeforeCopiesToHost(
                graph,
                tensor,
                getVirtualGraphIdOrSourceIpu(producer, tensor),
                producer->getPipelineStage());
          }
        }
      }
    }
  }
  return true;
}

RestoreOp *Pipeline::addNewRestoreOp(Graph &graph, int64_t stashSize) const {
  Op::Settings settings(graph, "");
  auto restoreOp_up = std::make_unique<RestoreOp>(
      Onnx::CustomOperators::Restore, stashSize, settings);
  auto restoreOp = restoreOp_up.get();
  graph.moveIntoGraph(std::move(restoreOp_up));

  return restoreOp;
}

RestoreInplaceOp *Pipeline::addNewRestoreInplaceOp(Graph &graph,
                                                   int64_t stashSize) const {
  Op::Settings settings(graph, "");
  auto restoreOp_up = std::make_unique<RestoreInplaceOp>(
      Onnx::CustomOperators::RestoreInplace, stashSize, settings);
  auto restoreOp = restoreOp_up.get();
  graph.moveIntoGraph(std::move(restoreOp_up));

  return restoreOp;
}

PipelineInfo::PipelineInfo(int64_t _batchesPerStep,
                           int64_t _gradAcclFactor,
                           int64_t _numPipelineStages,
                           bool _doGradAccl)
    : numStages(_numPipelineStages), doGradAccl(_doGradAccl) {

  auto fillFlushPhaseCycles = _numPipelineStages - 1;
  fillPhase.start           = 0;
  fillPhase.end             = fillFlushPhaseCycles - 1;

  int64_t mainCycles;
  if (doGradAccl) {
    mainCycles = _gradAcclFactor - fillFlushPhaseCycles;
  } else {
    mainCycles = _batchesPerStep - fillFlushPhaseCycles;
  }
  if (mainCycles < 1) {
    throw internal_error(
        "Pipeline mainCycles should not be less than 1. Current value is {}.",
        mainCycles);
  }

  mainPhase.start = fillPhase.end + 1;
  mainPhase.end   = mainPhase.start + mainCycles - 1;

  flushPhase.start = mainPhase.end + 1;
  flushPhase.end   = flushPhase.start + fillFlushPhaseCycles - 1;
}

bool PipelineInfo::doStage(PipelineCycle pCycle, PipelineStage pStage) const {
  bool doStageAtAll = pStage < numStages;
  bool doStageLower = (pCycle >= pStage);
  bool doStageUpper = (pCycle < pStage + flushPhase.start);

  return (doStageAtAll && doStageLower && doStageUpper);
}

namespace {
bool init = Transform::registerTransform(new Pipeline);
} // namespace

ExplicitPipelineHelper::ExplicitPipelineHelper(Graph &innerLoopSubgraph_)
    : innerLoopSubgraph(innerLoopSubgraph_), ir(innerLoopSubgraph.getIr()),
      pInfo(ir.pipelineInfo()),
      pipelineMainLoop(MainLoops::getInnerLoopOp(ir)) {}

void ExplicitPipelineHelper::createExplicitPipeline() {
  compatibilityChecks();
  ir.dotCheckpoint(ir, "ExplicitPipelineBeforeOutline");
  findOpsToOutline();
  createCallOps();
  ir.dotCheckpoint(ir, "ExplicitPipelineAfterOutline");
  decompose();
  cleanUp();
  ir.dotCheckpoint(ir, "ExplicitPipelineAfterDecompose");
}

std::pair<PipelineStage, PipelineStage>
ExplicitPipelineHelper::getMinAndMaxUnrollStages() const {
  // minStage == 1 means in the first unroll step, only stage 0 will be pulled
  // out of the LoopOp subgraph (and added before the LoopOp).
  PipelineStage minStage = 1;

  // maxStage == numStages - 1 means in the last unroll step, only the last
  // stage will be pulled out of the LoopOp subgraph
  // (and added behind the LoopOp).
  PipelineStage maxStage = pInfo.numStages - 1;

  // Check where (which PipelineStage) overlapped IO is required.
  auto overlapRequired = OverlapIO::overlapIORequired(ir);
  auto overlapLoops    = overlapRequired.find(ExchangeStrategy::OverlapLoops);
  auto overlapInnerLoop =
      overlapRequired.find(ExchangeStrategy::OverlapInnerLoop);

  // The lowest PipelineStage requiring an overlapped IO strategy
  OptionalPipelineStage minOverlapStage;

  // The highest PipelineStage requiring an overlapped IO strategy
  OptionalPipelineStage maxOverlapStage;

  if (overlapLoops != overlapRequired.end() &&
      overlapInnerLoop != overlapRequired.end()) {
    minOverlapStage = std::min(*(overlapLoops->second.begin()),
                               *(overlapInnerLoop->second.begin()));
    maxOverlapStage = std::max(*--(overlapLoops->second.end()),
                               *--(overlapInnerLoop->second.end()));
  } else if (overlapLoops != overlapRequired.end()) {
    minOverlapStage = *(overlapLoops->second.begin());
    maxOverlapStage = *--(overlapLoops->second.end());
  } else if (overlapInnerLoop != overlapRequired.end()) {
    minOverlapStage = *(overlapInnerLoop->second.begin());
    maxOverlapStage = *--(overlapInnerLoop->second.end());
  }

  // Overlapped IO on first stage required, which means one extra loop unroll
  if (minOverlapStage && *minOverlapStage == PipelineStage{0}) {
    // First unroll step will only pull IO out of the LoopOp subgraph
    minStage = 0;
  }

  // Overlapped IO on last stage required, which means one extra loop unroll
  if (maxOverlapStage &&
      *maxOverlapStage == PipelineStage{pInfo.numStages - 1}) {
    // Last unroll step will only pull IO out of the LoopOp subgraph
    maxStage = pInfo.numStages;
  }

  return {minStage, maxStage};
}

void ExplicitPipelineHelper::compatibilityChecks() const {
  // 1.
  if (!ir.getSessionOptions().useHostCopyOps) {
    throw error("SessionOption 'useHostCopyOps' must be true.");
  }

  // 2.
  if (!ir.getSessionOptions().enableExplicitMainLoops) {
    throw error("SessionOption 'enableExplicitMainLoops' must be true.");
  }

  // 3.
  if (!ir.getSessionOptions().explicitRecomputation) {
    throw error("SessionOption 'explicitRecomputation' must be true.");
  }

  // 4. Not compatible if EveryN anchor exists, as this adds an implicit
  //    counter incremented every PipelineCycle.
  if (ir.getDataFlow().isBatchCountingRequired()) {
    throw error("[Explicit Pipeline] Cannot execute when an anchor has "
                "AnchorReturnType 'EveryN'");
  }
}

void ExplicitPipelineHelper::findOpsToOutline() {
  opsToOutline.clear();

  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  auto minAndMaxStage = getMinAndMaxUnrollStages();

  // Reuse the same Op classification model as for loop unrolling
  DecomposeLoopPipelineModel model(minAndMaxStage.first,
                                   minAndMaxStage.second,
                                   pInfo.numStages,
                                   DecomposeTopoConLevel::None,
                                   DecomposeTopoConLevel::None,
                                   DecomposeTopoConLevel::None,
                                   computeLikeExchangeStrategies);

  auto classOps = model.classifyOperations(innerLoopSubgraph);

  for (auto &classOp : classOps) {
    auto utype = model.unwrap(classOp.second);
    // Only pipeline stage compute operations should be outlined at this stage
    if (utype.getType() == DecomposeLoopOpTypeEnum::Compute &&
        !utype.isPipelineIpuCopy()) {
      // One CallOp per pipeline stage
      opsToOutline[utype.getPipelineStage()].push_back(classOp.first->id);
    }
  }
}

void ExplicitPipelineHelper::createCallOps() {
  std::map<PipelineStage, Graph *> subgraphs;
  std::map<PipelineStage, OpId> pipelineStageAndOpId;

  AliasesMap aliasesMap{&ir};

  for (auto pstage_ops : opsToOutline) {
    auto pStage                = pstage_ops.first;
    auto ops                   = pstage_ops.second;
    auto subgraphableOpCluster = SubgraphableOpCluster(ops, &innerLoopSubgraph);
    std::map<Op *, int> index_map;
    std::string sgId = "PipelineStage_" + std::to_string(pStage) + "_main";
    auto &subgraph   = SubgraphOutline::createSubgraph(
        {subgraphableOpCluster}, ir, index_map, sgId);

    subgraphs.emplace(pStage, &subgraph);

    pipelineStageAndOpId[pStage] =
        SubgraphOutline::replaceWithCallOp(
            subgraphableOpCluster, subgraph, index_map, aliasesMap)
            ->id;
  }
}

void ExplicitPipelineHelper::decompose() {
  DecomposeLoops decomposer;

  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  // Unroll skewed by pipeline stage such that the main loop ends up
  // overlapping all pipeline stages
  DecomposeTopoConLevel before = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel loop   = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel after  = DecomposeTopoConLevel::Full;

  auto minAndMaxStage = getMinAndMaxUnrollStages();

  auto model = DecomposeLoopPipelineModel(minAndMaxStage.first,
                                          minAndMaxStage.second,
                                          pInfo.numStages,
                                          before,
                                          loop,
                                          after,
                                          computeLikeExchangeStrategies);

  decomposer.decomposeLoop(
      pipelineMainLoop->getGraph(), pipelineMainLoop, model);
}

void ExplicitPipelineHelper::cleanUp() {
  // Unset pipeline stage attributes of all ops, since this is no longer
  // required after the pipeline is represented explicitly in the IR
  for (auto op : ir.getAllOps()) {
    op->setPipelineStage({});
  }
}

} // namespace popart
