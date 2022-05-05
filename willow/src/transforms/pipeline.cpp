// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>

#include <popart/aliasesmap.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/loss.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/patterns/contiguateipucopyindices.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/mainloops.hpp>
#include <popart/transforms/pipeline.hpp>
#include <popart/transforms/randomsetup.hpp>
#include <popart/transforms/subgraphoutline.hpp>
#include <popart/util.hpp>
#include <popart/vertex.hpp>

#include <popart/graphutils.hpp>

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

void checkOpsPipelineStage(Graph &graph) {
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
std::vector<Op *> findDescendentsOnDifferentPipelineStages(PipelineInfo info,
                                                           Tensor *t,
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
    } else if (op->getPipelineStage() > stashRefOp->getPipelineStage() &&
               info.executeWithStage(op->getPipelineStage()) !=
                   info.executeWithStage(stashRefOp->getPipelineStage())) {
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
  } else if (tensor->isHostLoadTensor()) {
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

std::vector<Op *>
getRestoreReferenceOps(PipelineInfo info, Tensor *t, Op *stashRefOp) {
  logging::debug("Collecting restore ref candidates");
  auto consumers = t->consumers.getOps();

  std::vector<Op *> restoreCandidates;
  std::vector<PipelineStage> restorePipelineStages;
  for (auto c : consumers) {
    if (getVirtualGraphIdOrSourceIpu(c, t) ==
            getVirtualGraphIdOrSourceIpu(stashRefOp, t) &&
        c->getPipelineStage() != stashRefOp->getPipelineStage() &&
        info.executeWithStage(c->getPipelineStage()) !=
            info.executeWithStage(stashRefOp->getPipelineStage())) {
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

bool isFullRecompute(Graph &graph) {
  auto &ir = graph.getIr();
  return ir.canTrain() && (ir.getSessionOptions().autoRecomputation ==
                           RecomputationType::Pipeline);
}

bool hasCheckpointProducer(Tensor *tensor) {
  return !tensor->hasProducer() ||
         tensor->getProducer()->settings.recomputeType ==
             RecomputeType::Checkpoint;
}

bool onlyConsumedByPostLossOps(Tensor *tensor) {
  for (auto consumer : tensor->consumers.getOps()) {
    if (consumer->scheduledPreLoss == ScheduledPreLoss::Yes) {
      return false;
    }
  }
  return true;
}

std::set<TensorId> getStashCandidateTensors(Graph &graph) {
  auto info = graph.getIr().pipelineInfo();

  bool full_recompute = isFullRecompute(graph);

  std::set<TensorId> toStashCandidateTensors;
  for (auto &tid : graph.getTensors().getAllTensorIds()) {
    auto tensor = graph.getTensors().get(tid);

    if (tensor->consumers.getOps().empty() ||
        tensor->tensorType() == TensorType::Variable ||
        tensor->tensorType() == TensorType::Const ||
        tensor->isOptimizerTensor()) {
      continue;
    }

    // Full Recompute use stashes only on the inputs to an IPU
    // to complete any pipeline stage. Or tensors specified by the user.
    if (full_recompute && isProducedOnIPU(tensor) &&
        !hasCheckpointProducer(tensor)) {
      continue;
    }

    // We only concern ourselves with the normal context
    if (tensor->hasProducer() &&
        tensor->getProducer()->settings.executionContext !=
            popart::ExecutionContext::Normal) {
      continue;
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

    std::set<PipelineStage> tensorStagesCombined;
    for (auto stage : tensorStages) {
      tensorStagesCombined.insert(info.executeWithStage(stage));
    }

    // There is no need to stash a tensor that only appears in 1 stage.
    // Unless using full_recompute, then it must be consumed by something
    // other than a copy (it's not just "passing through"), which is
    // scheduled pre-loss (we're not recomputing grad ops)
    if ((tensorStages.size() == 1 || tensorStagesCombined.size() == 1) &&
        !(full_recompute && !onlyConsumedByCopies(tensor) &&
          !onlyConsumedByPostLossOps(tensor))) {
      continue;
    }

    logging::transform::debug("Adding {} to stash candidates", tid);
    toStashCandidateTensors.insert(tid);
  }

  return toStashCandidateTensors;
}

bool isRecomputable(Op *op) {
  if (op->settings.executionContext != ExecutionContext::Normal) {
    return false;
  }
  if (op->isConvertibleTo<HostLoadOp>()) {
    return false;
  }
  if (op->isConvertibleTo<InitOp>()) {
    return false;
  }

  // Copy ops are never recomputable
  if (op->isConvertibleTo<IpuCopyOp>()) {
    return false;
  }
  // Dont recompute the GetRandomSeedOp, or the identity that clones it.
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
  bool full_recompute = isFullRecompute(graph);

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
      }
    }
    return false;
  };

  // Initialise ops to be Recompute, except Ops whose output enters an IpuCopy,
  // or IpuCopys themselves.
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (isRecomputable(op)) {
      if (full_recompute) {
        // In full_recompute all forward ops are Recomputed unless specified by
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

} // namespace

std::map<PipelineStage, PipelineStage> Pipeline::withStages(const Ir &ir) {
  std::map<PipelineStage, PipelineStage> withStages;

  std::map<PipelineStage, std::set<VGraphId>> pipelineStageVGraphMap;

  const Graph *graph = &ir.getMainGraph();

  for (auto &op : graph->getOps()) {
    if ((op.second->settings.executionContext == ExecutionContext::Normal ||
         op.second->settings.executionContext == ExecutionContext::Subgraph) &&
        op.second->hasPipelineStage() && op.second->hasVirtualGraphId()) {
      pipelineStageVGraphMap[op.second->getPipelineStage()].insert(
          op.second->getVirtualGraphId());
      auto ps                                   = op.second->getPipelineStage();
      withStages[op.second->getPipelineStage()] = ps;
    }
  }

  PipelineStage stageOffset = 0;
  OptionalPipelineStage prevStage;
  std::set<VGraphId> prevVGraphIds;
  for (auto stageAndVGraphIds : pipelineStageVGraphMap) {
    if (prevStage) {
      std::set<VGraphId> result;
      std::set_intersection(stageAndVGraphIds.second.begin(),
                            stageAndVGraphIds.second.end(),
                            prevVGraphIds.begin(),
                            prevVGraphIds.end(),
                            std::inserter(result, result.begin()));
      if (!result.empty()) {
        // If the two sets of virtual graph IDs overlap, execute the current
        // pipeline stage with the previous one.
        withStages[stageAndVGraphIds.first] = withStages[*prevStage];
        ++stageOffset;
      } else {
        withStages[stageAndVGraphIds.first] =
            withStages[stageAndVGraphIds.first] - stageOffset;
      }
      logging::trace(
          "[Pipeline::withStages] Executing stage {} together with stage {}",
          stageAndVGraphIds.first,
          withStages[stageAndVGraphIds.first]);
    }

    prevStage     = stageAndVGraphIds.first;
    prevVGraphIds = stageAndVGraphIds.second;
  }

  return withStages;
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

  if (ir.getSessionOptions().explicitPipeliningEnabled()) {
    checkOpsPipelineStage(MainLoops::getInnerLoopSubgraph(ir));
  } else {
    checkOpsPipelineStage(graph);
  }

  // 2. Currently user-annotated recomputation is not supported with pipelining
  // (TODO T9575)
  if (ir.getMainGraph().hasUserRecomputeOps()) {
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

  auto getIpuCopyOps = [&graph] {
    // contiguating IpuCopyOps
    std::vector<popart::IpuCopyOp *> ipuCopies;
    for (auto &op_pair : graph.getOps()) {
      auto ipuCopyOp = dynamic_cast<popart::IpuCopyOp *>(op_pair.second.get());
      if (ipuCopyOp &&
          ipuCopyOp->settings.executionContext == ExecutionContext::Normal) {
        ipuCopies.push_back(ipuCopyOp);
      }
    }
    return ipuCopies;
  };

  chainCopiesTransform(graph);

  // Other sharding assumptions to check:

  // 4. Ir stream tensors cannot be consumed by ops on multiple IPUs
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

  // Now apply the transform

  // 0. Contiguate the IPUCopies
  ContiguateIpuCopyIndicesPattern contiguator;
  for (auto ipuCopyOp : getIpuCopyOps()) {
    if (!ipuCopyOp->isExcludedFromPattern(&contiguator) &&
        contiguator.matches(ipuCopyOp)) {
      logging::transform::debug("Contiguating {}", ipuCopyOp->debugName());
      contiguator.apply(ipuCopyOp);
    }
  }
  ir.updateVertices();

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

  if (ir.getSessionOptions().explicitPipeliningEnabled()) {
    // Get the inner loop subgraph that is to be pipelined
    auto &toPipelineGraph = MainLoops::getInnerLoopSubgraph(ir);
    addStashRestoreOps(toPipelineGraph);
    return applyExplicit(toPipelineGraph);
  } else {
    // The graph to be pipelined is always the main graph
    return addStashRestoreOps(graph);
  }
}

int Pipeline::getStashSize(const Ir &ir,
                           PipelineStage stashStage,
                           PipelineStage maxRestoreStage) const {
  return ir.pipelineInfo().numIndependentStages(stashStage,
                                                maxRestoreStage + 1);
}

bool Pipeline::applyExplicit(Graph &innerLoopSubgraph) const {
  ExplicitPipelineHelper explicitPipeline(innerLoopSubgraph);
  explicitPipeline.createExplicitPipeline();
  return true;
}

bool Pipeline::addStashRestoreOps(Graph &graph) const {
  auto &ir            = graph.getIr();
  bool full_recompute = isFullRecompute(graph);
  auto pipelineInfo   = ir.pipelineInfo();

  auto toStashCandidateTensors = getStashCandidateTensors(graph);

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

  std::set<TensorId> toStashTensors;
  // StashTensorId -> std::pair<StashRefOp, RestoreRefOps>
  std::map<TensorId, std::pair<Op *, std::vector<Op *>>> stashRestoreRefOps;
  // If there is no recomputation, then the candidates for stashing will all be
  // stashed.
  if (!ir.autoRecomputationEnabled()) {
    toStashTensors = toStashCandidateTensors;
  }

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
  else {
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
        // For full_recompute if a stash candidate doesn't have a
        // restoreReference then it is not required for recomputation during the
        // backwards pass.
        if (full_recompute) {
          auto stashRef    = getStashReferenceOp(tensor);
          auto restoreRefs = findDescendentsOnDifferentPipelineStages(
              pipelineInfo, tensor, stashRef);
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

  logging::transform::debug("Final Stash Tensors");
  for (auto tid : toStashTensors) {
    logging::transform::debug("  {}", tid);
  }

  Op::Settings settings(graph, "");

  for (auto &tid : toStashTensors) {
    auto tensor = graph.getTensors().get(tid);

    if (tensor->consumers.getOps().empty()) {
      throw internal_error("request to stash Tensor {} with no "
                           "consumers, bailing",
                           tensor->str());
    }

    Op *stashRefOp;
    std::vector<Op *> restoreRefOps;
    if (stashRestoreRefOps.find(tid) != stashRestoreRefOps.end()) {
      auto refs     = stashRestoreRefOps.at(tid);
      stashRefOp    = refs.first;
      restoreRefOps = refs.second;
    } else {
      stashRefOp    = getStashReferenceOp(tensor);
      restoreRefOps = getRestoreReferenceOps(pipelineInfo, tensor, stashRefOp);
    }

    std::vector<PipelineStage> restoreStages;
    for (Op *restoreRefOp : restoreRefOps) {
      restoreStages.push_back(restoreRefOp->getPipelineStage());
    }

    // The largest PipelineStage of the restoreRefOps determines stash size
    auto stashSize = getStashSize(
        ir,
        stashRefOp->getPipelineStage(),
        *std::max_element(restoreStages.begin(), restoreStages.end()));

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
    if (full_recompute) {
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
  if (full_recompute) {
    for (auto &tid : graph.getTensors().getAllTensorIds()) {
      auto tensor = graph.getTensors().get(tid);

      // Stash tensors have already been covered above
      if (std::find(toStashTensors.cbegin(), toStashTensors.cend(), tid) !=
          toStashTensors.cend()) {
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
                           bool _doGradAccl,
                           std::map<PipelineStage, PipelineStage> _withStage)
    : numStages(_numPipelineStages), doGradAccl(_doGradAccl),
      withStage(_withStage) {

  auto fillFlushPhaseCycles = numIndependentStages(0, numStages) - 1;
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

PipelineStage PipelineInfo::executeWithStage(PipelineStage pStage) const {
  auto it = withStage.find(pStage);
  return it == withStage.end() ? pStage : it->second;
}

int PipelineInfo::numIndependentStages(PipelineStage start, PipelineStage end) {
  std::set<PipelineStage> stages;
  for (PipelineStage stage = start; stage < end; ++stage) {
    auto it = withStage.find(stage);
    stages.insert(it == withStage.end() ? stage : it->second);
  }
  return stages.size();
}

bool PipelineInfo::doStage(PipelineCycle pCycle, PipelineStage pStage) const {
  logging::trace(
      "[ PipelineInfo::doStage] Executing stage {} together with stage {}",
      pStage,
      executeWithStage(pStage));
  bool doStageAtAll = pStage < numStages;
  pStage            = executeWithStage(pStage);
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
  categorizeInnerLoopOps();
  createCallOps();

  // Reduce the trip count of the loop op to account for the extra fill and
  // flush calls
  pipelineMainLoop->setTripCountValue(pInfo.getMainCycles());

  // In addition, we need to clone the innerLoopSubgraph as we will loop over
  // the scheduled ops and modify it in the unrolling
  cloneSrct.originalGraphOpIdAndClonedGraphOpId =
      ir.cloneGraph(innerLoopSubgraph.id, {cloneSrct.clonedGraphId}).opIdMap;
  for (const auto originalAndClonedOp :
       cloneSrct.originalGraphOpIdAndClonedGraphOpId) {
    cloneSrct.clonedGraphOpIdAndOriginalGraphOpId[originalAndClonedOp.second] =
        originalAndClonedOp.first;
  }

  // Unroll the loop for the fill stage
  auto lastOriginalAndClonedTensorId = createFillPhase();
  auto tensorIdsToFlushStage =
      modifyInputAndOutputInInnerLoop(lastOriginalAndClonedTensorId);
  createFlushPhase(tensorIdsToFlushStage);

  cleanUp();
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

void ExplicitPipelineHelper::categorizeInnerLoopOps() {
  auto isForStreamingFromHost = [](Op *op) {
    if (op->isConvertibleTo<HostLoadOp>()) {
      return true;
    }
    // If an InitOp whose output is consumed by a HostLoadOp ??
    if (op->isConvertibleTo<InitOp>()) {
      auto consumerOps =
          op->outTensor(InitOp::getOutIndex())->consumers.getOps();
      if (consumerOps.size() == 1 &&
          consumerOps[0]->isConvertibleTo<HostLoadOp>()) {
        return true;
      }
    }
    return false;
  };

  auto insertOp = [](Op *op,
                     std::map<PipelineStage, std::vector<OpId>> &pStageOpIds) {
    if (op->hasPipelineStage()) {
      PipelineStage pStage = op->getPipelineStage();

      auto it = pStageOpIds.find(pStage);
      if (it != pStageOpIds.end()) {
        it->second.push_back(op->id);
      } else {
        pStageOpIds.emplace(pStage, std::vector<OpId>{op->id});
      }
    } else {
      throw error("[Explicit Pipeline] Op {} does not have the PipelineStage "
                  "attribute set",
                  op->str());
    }
  };

  for (OpId opId : innerLoopSubgraph.getOpIds()) {
    Op *op = innerLoopSubgraph.getOp(opId);
    if (op->isIpuCopyOp()) {
      insertOp(op, innerLoopOpsCategories.ipuCopyOps);
    } else if (isForStreamingFromHost(op)) {
      insertOp(op, innerLoopOpsCategories.hostLoadOps);
    } else if (op->isConvertibleTo<HostStoreOp>()) {
      insertOp(op, innerLoopOpsCategories.hostStoreOps);
    } else {
      insertOp(op, innerLoopOpsCategories.mainOps);
    }
  }
}

std::map<PipelineStage, OpId>
ExplicitPipelineHelper::replaceOpsCategoriesWithCallOps(
    const std::map<PipelineStage, std::vector<OpId>> &opsCategory,
    std::string subgraphPostfix) {
  std::map<PipelineStage, Graph *> subgraphs;
  std::map<PipelineStage, OpId> pipelineStageAndOpId;

  AliasesMap aliasesMap{&ir};

  for (auto pstage_ops : opsCategory) {
    auto pStage                = pstage_ops.first;
    auto ops                   = pstage_ops.second;
    auto subgraphableOpCluster = SubgraphableOpCluster(ops, &innerLoopSubgraph);
    std::map<Op *, int> index_map;
    std::string sgId =
        "PipelineStage_" + std::to_string(pStage) + "_" + subgraphPostfix;
    auto &subgraph = SubgraphOutline::createSubgraph(
        {subgraphableOpCluster}, ir, index_map, sgId);

    subgraphs.emplace(pStage, &subgraph);

    pipelineStageAndOpId[pStage] =
        SubgraphOutline::replaceWithCallOp(
            subgraphableOpCluster, subgraph, index_map, aliasesMap)
            ->id;
  }
  return pipelineStageAndOpId;
}

void ExplicitPipelineHelper::createCallOps() {
  auto mainCallOps =
      replaceOpsCategoriesWithCallOps(innerLoopOpsCategories.mainOps, "Main");
  auto hostLoadCallOps = replaceOpsCategoriesWithCallOps(
      innerLoopOpsCategories.hostLoadOps, "FromHost");
  auto hostStoreCallOps = replaceOpsCategoriesWithCallOps(
      innerLoopOpsCategories.hostStoreOps, "ToHost");
  pipelineStageOpIdMaps.ipuCopyCallOps = replaceOpsCategoriesWithCallOps(
      innerLoopOpsCategories.ipuCopyOps, "IpuCopy");

  // As we are going to loop over all ops in a scheduled order we create a map
  // of all inner loop ops and their corresponding pipeline stage
  for (auto pStageOps : mainCallOps) {
    pipelineStageOpIdMaps.opIdAndPipelineStage[pStageOps.second] =
        pStageOps.first;
  }
  for (auto pStageOps : hostLoadCallOps) {
    pipelineStageOpIdMaps.opIdAndPipelineStage[pStageOps.second] =
        pStageOps.first;
  }
  for (auto pStageOps : hostStoreCallOps) {
    pipelineStageOpIdMaps.opIdAndPipelineStage[pStageOps.second] =
        pStageOps.first;
  }
  for (auto pStageOps : pipelineStageOpIdMaps.ipuCopyCallOps) {
    pipelineStageOpIdMaps.opIdAndPipelineStage[pStageOps.second] =
        pStageOps.first;
  }
}

std::map<TensorId, TensorId> ExplicitPipelineHelper::createFillPhase() {
  // Get graph which piplineMainLoop is part of
  auto &pipelineMainLoopOuterGraph = pipelineMainLoop->getGraph();

  // Map to keep track of which tensor id from the original set of tensors will
  // map to cloned tensors
  std::map<TensorId, TensorId> lastOriginalAndClonedTensorId;

  // Get the operators in correct order so that we can add tensors as consumers
  // to cloned ops
  // Don't need the optimal schedule as any valid order would suffice to get the
  // tensors in the correct order
  auto scheduledOps =
      innerLoopSubgraph.getOpSchedule({}, RequireOptimalSchedule::No);

  // Nested loop where the outer loop sets the limit on the maximum
  // PipelineStage which is allowed in an operator when creating the fill stage
  // In other words: We will unroll the loop until the pipelinestage max
  // Note that the fill stage consist of one less pipeline than the number of
  // pipeline stages
  for (auto pStageMax = pInfo.fillPhase.end; pStageMax >= pInfo.fillPhase.start;
       pStageMax--) {
    // As we need a new TensorID for each IPU which are part of the fill stage
    // we will create a map between the original tensors (i.e. those in the
    // original loop operator) and the corresponding tensor ids in the unrolled
    // loop
    std::map<TensorId, TensorId> originalTensorIdAndNewIntermediateTensorId;

    for (auto op : scheduledOps) {
      if (pipelineStageOpIdMaps.opIdAndPipelineStage.count(op->id) == 0) {
        // The operator is not part of any pipeline stages
        continue;
      }
      auto pStage = pipelineStageOpIdMaps.opIdAndPipelineStage.at(op->id);
      if (pStage > pStageMax) {
        // The pipeline stage is higher than the number of cycles used for
        // filling
        continue;
      }

      // Clone the operator
      auto clonedOpUp = op->clone();

      // Change ownership of the cloned operator after obtaining the raw pointer
      auto clonedOp = clonedOpUp.get();
      pipelineMainLoopOuterGraph.moveIntoGraph(std::move(clonedOpUp));

      // Change scope of the clonedOp so that it is no longer a part of the
      // innerLoop
      clonedOp->settings.scope = pipelineMainLoop->settings.scope;

      // The cloned operator should not be connected to any tensors of the
      // original operator (as these will belong to the inner loop) New tensors
      // (belonging to the fill phase) replacing these (belonging to the inner
      // loop) will be created below
      clonedOp->disconnectAllInputs();
      clonedOp->disconnectAllOutputs();

      // First we clone the input tensors
      auto tensorInputMap = op->input->tensorMap();
      for (auto indexAndTensor : tensorInputMap) {
        auto index  = indexAndTensor.first;
        auto tensor = indexAndTensor.second;

        // We remove the inner loop scope from the tensor
        auto newInputTensorId = removeScope(innerLoopSubgraph, tensor->id);

        if (tensor->hasProducer()) {
          // As we are looping over the ops schedule the first tensor will not
          // have a producer
          newInputTensorId =
              originalTensorIdAndNewIntermediateTensorId.at(tensor->id);
        }

        // Attach to the new tensor to the cloned op
        clonedOp->connectInTensor(index, newInputTensorId);
      }
      // Then we clone the output tensors
      auto tensorOuputMap = op->output->tensorMap();
      for (auto indexAndTensor : tensorOuputMap) {
        auto index  = indexAndTensor.first;
        auto tensor = indexAndTensor.second;

        // We remove the inner loop scope from the tensor
        auto newOutputTensorId = removeScope(innerLoopSubgraph, tensor->id);
        // Create the tensor with the tensorId made above

        // It could be tempting to set the tensorIds with graph.addScope()
        // However, we will be creating these output several times (one for each
        // pipeline stage minus one) Thus, we have to create an intermediate
        // tensor (i.e. one which is not created by the end-user)
        auto newIntermediateOutputTensorId =
            ir.createIntermediateTensorId(newOutputTensorId);
        clonedOp->createAndConnectOutTensor(index,
                                            newIntermediateOutputTensorId);

        originalTensorIdAndNewIntermediateTensorId[tensor->id] =
            newIntermediateOutputTensorId;

        // Update the map of original and intermediate tensors
        if (pStage == pStageMax) {
          lastOriginalAndClonedTensorId[tensor->id] =
              newIntermediateOutputTensorId;
        }
      }
      // Propagate tensor info
      clonedOp->setup();
    }
  }

  return lastOriginalAndClonedTensorId;
}

std::map<std::pair<PipelineStage, TensorId>, TensorId>
ExplicitPipelineHelper::modifyInputAndOutputInInnerLoop(
    const std::map<TensorId, TensorId> lastOriginalAndClonedTensorId) {
  // Add the input from the fill pahse to the main phase of the LoopOp

  // Extract the output from the main phase which will be used in the flush
  // phase
  std::map<std::pair<PipelineStage, TensorId>, TensorId> tensorIdsToFlushStage;

  for (auto pipelineStageAndOpId : pipelineStageOpIdMaps.ipuCopyCallOps) {
    auto pStage = pipelineStageAndOpId.first;
    auto opId   = pipelineStageAndOpId.second;

    // Loop over tensors belonging to the opId of the current pipelineStage
    Op *op              = innerLoopSubgraph.getOp(opId);
    auto tensorOuputMap = op->output->tensorMap();
    for (auto indexAndTensor : tensorOuputMap) {
      auto outTensor = indexAndTensor.second;

      // Add tensors from the fill phase to the loopOp
      // An intermediate (i.e. not user specified) tensor for the output from
      // the current iteration (which is going to be the input of the next
      // iteration)
      auto loopInId = ir.createIntermediateTensorId(outTensor->id);
      // Add the tensor which is an output of the copy operator as the input for
      // the next iteration
      pipelineMainLoop->addLoopInput(
          LoopOp::getFirstInputInIndex(),
          lastOriginalAndClonedTensorId.at(outTensor->id),
          loopInId,
          false);

      // Add output which are going to the flush stage
      auto loopOutId = removeScope(innerLoopSubgraph, outTensor->id);
      // The output in this pipeline stage in the main loop will be connected to
      // the next pipeline stage in the flush stage, thus we add 1 to pStage
      tensorIdsToFlushStage[{pStage + 1, outTensor->id}] = loopOutId;
      pipelineMainLoop->addLoopOutput(
          LoopOp::getFirstOutputOutIndex(), loopOutId, outTensor->id, false);

      // Disconnect the input tensor (which is the output tensor of the copy op)
      // of the ops consuming the output tensor The data in the tensor will be
      // loop carried as described above
      auto outTensorConsumerOps = outTensor->consumers.getOps();
      for (auto outTensorConsumerOp : outTensorConsumerOps) {
        auto indices = outTensorConsumerOp->input->indices(outTensor);
        for (auto index : indices) {
          outTensorConsumerOp->disconnectInTensor(index);
          outTensorConsumerOp->connectInTensor(index, loopInId);
        }
      }
    }
  }

  // Setup the main loop after altering it
  pipelineMainLoop->setup();

  return tensorIdsToFlushStage;
}

void ExplicitPipelineHelper::createFlushPhase(
    const std::map<std::pair<PipelineStage, TensorId>, TensorId>
        tensorIdsToFlushStage) {
  // Get graph which piplineMainLoop is part of
  auto &pipelineMainLoopOuterGraph = pipelineMainLoop->getGraph();
  // Get the cloned innerLoopSubgraph
  auto &innerLoopSubgraphClone = ir.getGraph({cloneSrct.clonedGraphId});

  // As we have altered the graph we need to loop over the shedule ops of the
  // cloned graph
  auto clonedScheduledOps =
      innerLoopSubgraphClone.getOpSchedule({}, RequireOptimalSchedule::No);

  // Nested loop where the outer loop sets the limit on the minimum
  // PipelineStage which is allowed in an operator when creating the flush stage
  // In other words: We will unroll the loop from pipelinestage min
  // Note that the flush stage consist of one less pipeline than the number of
  // pipeline stages
  for (auto pStageMin =
           1; // All but the first pipeline stage is part of the flush phase
       pStageMin <=
       pInfo.mainPhase.start; // There are as many flush stages as fill stages
       pStageMin++) {
    // As we need a new TensorID for each IPU which are part of the flush stage
    // we will create a map between the original tensors (i.e. those in the
    // original loop operator) and the corresponding tensor ids in the unrolled
    // loop
    std::map<TensorId, TensorId> originalTensorIdAndNewIntermediateTensorId;

    for (auto op : clonedScheduledOps) {
      if (cloneSrct.clonedGraphOpIdAndOriginalGraphOpId.count(op->id) == 0) {
        // The operator is not part of the cloned graph
        continue;
      }
      if (pipelineStageOpIdMaps.opIdAndPipelineStage.count(
              cloneSrct.clonedGraphOpIdAndOriginalGraphOpId.at(op->id)) == 0) {
        // The operator is not part of any pipeline stages
        continue;
      }
      auto pStage = pipelineStageOpIdMaps.opIdAndPipelineStage.at(
          cloneSrct.clonedGraphOpIdAndOriginalGraphOpId.at(op->id));
      if (pStage < pStageMin) {
        // The pipeline stage is lower than the number of cycles used for
        // flushing
        continue;
      }

      // Clone the operator
      auto clonedOpUp = op->clone();

      // Change ownership of the cloned operator after obtaining the raw pointer
      auto clonedOp = clonedOpUp.get();
      pipelineMainLoopOuterGraph.moveIntoGraph(std::move(clonedOpUp));

      // Change scope of the clonedOp so that it is no longer a part of the
      // innerLoop
      clonedOp->settings.scope = pipelineMainLoop->settings.scope;

      // The cloned operator should not be connected to any tensors of the
      // original operator (as these will belong to the inner loop) New tensors
      // (belonging to the flush phase) replacing these (belonging to the inner
      // loop) will be created below
      clonedOp->disconnectAllInputs();
      clonedOp->disconnectAllOutputs();

      // First we clone the input tensors
      auto tensorInputMap = op->input->tensorMap();
      for (auto indexAndTensor : tensorInputMap) {
        auto index  = indexAndTensor.first;
        auto tensor = indexAndTensor.second;

        // We remove the inner loop scope from the tensor
        auto newInputTensorId = removeScope(innerLoopSubgraphClone, tensor->id);

        auto mapIt = tensorIdsToFlushStage.find(
            {pStageMin, addScope(innerLoopSubgraph, newInputTensorId)});
        if (mapIt != tensorIdsToFlushStage.end()) {
          // The tensors is an output from the loop op
          newInputTensorId = mapIt->second;
        } else {
          // The input of this op comes from a newly cloned op
          newInputTensorId =
              originalTensorIdAndNewIntermediateTensorId.at(tensor->id);
        }

        // Attach to the new tensor to the cloned op
        clonedOp->connectInTensor(index, newInputTensorId);
      }
      // Then we clone the output tensors
      auto tensorOuputMap = op->output->tensorMap();
      for (auto indexAndTensor : tensorOuputMap) {
        auto index  = indexAndTensor.first;
        auto tensor = indexAndTensor.second;

        // We remove the inner loop scope from the tensor
        auto newOutputTensorId =
            removeScope(innerLoopSubgraphClone, tensor->id);
        // Create the tensor with the tensorId made above

        // It could be tempting to set the tensorIds with graph.addScope()
        // However, we will be creating these output several times (one for each
        // pipeline stage minus one) Thus, we have to create an intermediate
        // tensor (i.e. one which is not created by the end-user)
        auto newIntermediateOutputTensorId =
            ir.createIntermediateTensorId(newOutputTensorId);

        clonedOp->createAndConnectOutTensor(index,
                                            newIntermediateOutputTensorId);

        originalTensorIdAndNewIntermediateTensorId[tensor->id] =
            newIntermediateOutputTensorId;
      }
      // Propagate tensor info
      clonedOp->setup();
    }
  }
}

void ExplicitPipelineHelper::cleanUp() {
  // Unset pipeline stage attributes of all ops
  for (auto op : ir.getAllOps()) {
    op->setPipelineStage({});
  }

  ir.removeGraph({cloneSrct.clonedGraphId});
}

} // namespace popart
