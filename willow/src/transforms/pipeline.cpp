#include <vector>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/patterns/contiguateipucopyindices.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/pipeline.hpp>

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
//  - In-place the activation tensors when restoring
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

VGraphId getVirtualGraphIdOrSourceIpu(Op *op) {
  if (op->isConvertibleTo<IpuCopyOp>()) {
    auto ipuCopyOp = dynamic_cast<popart::IpuCopyOp *>(op);
    return static_cast<int64_t>(ipuCopyOp->getSourceIpu());
  } else {
    return op->getVirtualGraphId();
  }
}

void setCopyOpsPipelineStage(IpuCopyOp *op) {
  // Copies of optimizer tensors do not run in the main program fragment and
  // should not have their pipeline stage set.
  if (op->copiesOptimizerTensors()) {
    return;
  }

  auto in0 = op->inTensor(0);
  if (in0->hasProducer()) {
    auto producer = in0->getProducer();
    auto ps       = producer->getPipelineStage();
    op->setPipelineStage(ps);
  } else {
    throw error("Can not copy variable tensor {} between virtual graphs when "
                "pipelining. All pipeline stages using this tensor should be "
                "on the same graph.",
                in0->str());
  }
}

void checkOpsPipelineStage(Graph &graph) {
  // return the pipeline stage or -1 if not set
  // throw error if pipeline stage if negative
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

  // collect all ops in  each pipeline stage
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

  // use the pipeline stage of the source producer as the pipeline stage for the
  // IpuCopy
  logging::debug("Setting the pipeline stage attribute of the Ipu copy ops");
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->isConvertibleTo<IpuCopyOp>()) {
      auto copyOp = dynamic_cast<IpuCopyOp *>(op);
      setCopyOpsPipelineStage(copyOp);
    }
  }
}

} // namespace

bool Pipeline::apply(Graph &graph) const {

  auto &ir         = graph.getIr();
  auto maxVGraphId = ir.getMaxVirtualGraphId();
  // We use numIPUs // replicated graph count for the max vGraph ID.

  // First, some checks that pipelining is compatible with other user options:

  // 1. Pipelining uses the virtual graph API. This must be enabled
  if (!ir.virtualGraphsEnabled()) {
    throw error("Pipelining requires the 'virtualGraphMode' session option "
                "to not be VirtualGraphMode::Off.");
  }

  checkOpsPipelineStage(graph);

  // 2. There must be enough batches of data for the cycle of filling
  //    and flushing the pipeline
  int minDepth;
  if (ir.canTrain()) {
    minDepth = 2 * (maxVGraphId - 1) + 1;
  } else {
    minDepth = maxVGraphId;
  }

  int64_t depth;
  if (ir.getSessionOptions().enableGradientAccumulation) {
    depth = ir.getSessionOptions().accumulationFactor;
    if (depth < minDepth) {
      // For replicated graphs we are replicating the entire pipeline, so these
      // condidtions still hold.
      throw error("For pipelining, depth (gradient accumulation factor) must "
                  "be at least {} "
                  "for {} IPUs",
                  minDepth,
                  ir.getDeviceInfo()->getNumIpus());
    }
  } else {
    depth = ir.getDataFlow().batchesPerStep();
    if (depth < minDepth) {
      throw error("For pipelining, depth (batchesPerStep) must be at least {} "
                  "for {} IPUs",
                  minDepth,
                  ir.getDeviceInfo()->getNumIpus());
    }
  }

  // 3. Currently recomputation is not supported with pipelining (TODO T9575)
  if (ir.getMainGraph().hasUserRecomputeOps()) {
    throw error("When pipelining is enabled, user annotation for recomputation "
                "is not allowed");
  }

  // 4. Forward layers must be sharded with increasing IPU index
  //    Examples violating this:
  //      Consider the fwd Graph : Op0 -> Op1 -> Op2 -> Op3
  //          e.g. 1) IPU0 : {Op2, Op3}, IPU1 : {Op0, Op1}
  //          e.g. 2) IPU0 : {Op0, Op2}, IPU1 : {Op1, Op3}

  // The checks:
  // 4.2 Copies in the correct direction

  auto getIpuCopyOps = [&graph] {
    // contiguating IpuCopyOps
    std::vector<popart::IpuCopyOp *> ipuCopies;
    for (auto &op_pair : graph.getOps()) {
      auto ipuCopyOp = dynamic_cast<popart::IpuCopyOp *>(op_pair.second.get());
      if (ipuCopyOp) {
        ipuCopies.push_back(ipuCopyOp);
      }
    }
    return ipuCopies;
  };

  // Other sharding assumptions to check:

  // 5. Ir stream tensors cannot be consumed by ops on multiple IPUs
  for (TensorId tid : graph.getTensors().getIds(TensorType::Stream)) {
    auto tensor = graph.getTensors().get(tid);
    std::set<PipelineStage> pipelineStages;
    for (auto c : tensor->consumers.getOps()) {
      if (!c->isConvertibleTo<IpuCopyOp>() ||
          !dynamic_cast<IpuCopyOp *>(c)->copiesOptimizerTensors()) {
        pipelineStages.insert(c->getPipelineStage());
      }
    }

    if (pipelineStages.size() > 1) {
      throw error("For pipelining, stream tensors can only be streamed "
                  "directly onto a single IPU");
    }
  }

  // Now apply the transform

  // 0. Contiguate the IPUCopies
  ContiguateIpuCopyIndicesPattern contiguator;
  for (auto ipuCopyOp : getIpuCopyOps()) {
    if (contiguator.matches(ipuCopyOp)) {
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
        ss << fmt::format(
            "ILE: IpuCopy {} is not contiguous. It copies from IPU {} to "
            "IPU {}. Failed to contiguate all IpuCopyOps",
            ipuCopyOp->debugName(),
            sourceIpu,
            destIpu);
        ss << fmt::format("\nin tensor 0: {}", ipuCopyOp->inTensor(0)->str());
        ss << fmt::format("\nin tensor 0 producer pipeline stage: {}",
                          sourceIpu);
        ss << fmt::format("\nout tensor 0: {}", ipuCopyOp->outTensor(0)->str());
        ss << fmt::format("\nout tensor 0 lowest consumer pipeline stage: {}",
                          destIpu);
        throw error(ss.str());
      }
    }
  }

  if (!ir.canTrain()) {
    // No stashing of forward activations required in inference/eval mode
    return true;
  }

  // 1. We will insert topological constraints to ensure that relative positions
  // of Stash and Restore Ops w.r.t. Loss are correct. Finding the first Op with
  // a path from the Loss
  auto currentSchedule = graph.getOpSchedule({});
  auto firstFromLoss =
      std::find_if(currentSchedule.cbegin(),
                   currentSchedule.cend(),
                   [](Op *op) { return op->fromLoss == PathFromLoss::Yes; });
  if (firstFromLoss == currentSchedule.cend()) {
    throw error(
        "ILE: no Op with PathFromLoss::Yes, yet canTrain() is true, bailing");
  }
  logging::transform::debug("First PathFromLoss::Yes in schedule is {}.",
                            (*firstFromLoss)->str());

  // Ops which have no path to or from the loss can be scheduled pre- or post-
  // loss. In the pipelining transformation, the position of an Op relative
  // the loss is used to determine where and when to stash and restore Tensors.
  // Therefore, the position of all Ops relative to the loss must be fixed and
  // known. We here freeze positions relative to the loss.
  //
  // Note that for most situations, these constraints are redundant, as
  // the constraints added later (Stash before loss, Restore after loss) imply
  // them. But, there are edge cases where there is no Restore-Stash combo to
  // implicitly constrain this order (for recomputation)
  //
  bool beforeFirstFromLoss{true};
  for (auto op : currentSchedule) {
    if (op == *firstFromLoss) {
      beforeFirstFromLoss = false;
    } else if (op->toLoss == PathToLoss::No &&
               op->fromLoss == PathFromLoss::No) {
      if (beforeFirstFromLoss) {
        graph.topoCons->insert(op, *firstFromLoss);
      } else {
        graph.topoCons->insert(*firstFromLoss, op);
      }
    }
  }

  // 1. Find all tensors in the fwd pass that are inputs to ops in the bwd pass
  std::vector<TensorId> toStashCandidateTensors;
  for (auto &tid : graph.getTensors().getAllTensorIds()) {
    auto tensor = graph.getTensors().get(tid);

    // Not a candidate for stashing if the tensor:
    // - has no consumers
    // - is a variable tensor
    // - is an optimizer tensor
    // - is on the final IPU

    if (tensor->consumers.getOps().empty()) {
      continue;
    }
    if (tensor->tensorType() == TensorType::Variable) {
      continue;
    }
    if (tensor->isOptimizerTensor()) {
      continue;
    }

    auto pipelineStageCheckOp = tensor->consumers.getOps()[0];
    int pipelineStage =
        static_cast<int>(pipelineStageCheckOp->getPipelineStage());
    if (pipelineStage == maxVGraphId - 1) {
      continue;
    }

    bool isConsumedByOpScheduledPreLoss  = false;
    bool isConsumedByOpScheduledPostLoss = false;
    for (Op *consumer : tensor->consumers.getOps()) {
      if (consumer->scheduledPreLoss == ScheduledPreLoss::Yes) {
        isConsumedByOpScheduledPreLoss = true;
      } else if (consumer->scheduledPreLoss == ScheduledPreLoss::No) {
        isConsumedByOpScheduledPostLoss = true;
      }
    }

    bool isProducedPreLoss =
        tensor->hasProducer() &&
        tensor->getProducer()->scheduledPreLoss == ScheduledPreLoss::Yes;

    if ((isConsumedByOpScheduledPreLoss || isProducedPreLoss) &&
        isConsumedByOpScheduledPostLoss) {
      toStashCandidateTensors.push_back(tid);
    }
  }

  std::vector<TensorId> toStashTensors;
  // If there is no recomputation, then the candidates for stashing will all be
  // stashed.
  if (!ir.autoRecomputationEnabled()) {
    toStashTensors = toStashCandidateTensors;
  }

  // If there is recomputation, the candidate set is reduced.
  //
  // Candidate Tensors which can be recomputed from other stashing candidates,
  // are filtered out, and their producers are set to RECOMPUTE.
  //
  // The only exceptions are candidate stashing Tensors which are copied to
  // another IPU : these must be stashed even if they recomputable. This
  // guarantees that the correct Tensor is copied after fwd and bwd have
  // executed.
  //
  // Algorithm : initialize all pre-loss Ops to be RECOMPUTE, and then set to
  // CHECKPOINT if (1) cannot be computed from previous Stashed Tensors or (2)
  // must be copied to next IPU.
  else {

    // Initialise forward Ops to be Recompute, except Ops whose output enters an
    // IpuCopy.
    for (auto op : graph.getOpSchedule({})) {
      if (!dynamic_cast<IpuCopyOp *>(op) &&
          op->scheduledPreLoss == ScheduledPreLoss::Yes) {
        op->settings.recomputeType = RecomputeType::RECOMPUTE;
        for (auto tensor : op->output->tensors()) {
          for (auto consumer : tensor->consumers.getOps()) {
            if (dynamic_cast<IpuCopyOp *>(consumer)) {
              op->settings.recomputeType = RecomputeType::CHECKPOINT;
            }
          }
        }
      }
    }

    logging::transform::debug(
        "Reducing the set of stashing candidate Tensors for recomputation");

    // Finding initial set of Tensors which are not produced on their IPUs and
    // are not stashed
    std::vector<Tensor *> frontier;
    std::set<TensorId> beenOnFrontier;
    for (auto tid : graph.getTensors().getAllTensorIds()) {
      Tensor *tensor = graph.getTensors().get(tid);
      // not produced on IPU : stream tensor or copied on
      if ((!tensor->hasProducer() &&
           tensor->tensorType() == TensorType::Stream) ||
          (tensor->hasProducer() &&
           dynamic_cast<IpuCopyOp *>(tensor->getProducer()))) {
        // not stashed
        if (std::find(toStashCandidateTensors.cbegin(),
                      toStashCandidateTensors.cend(),
                      tensor->id) == toStashCandidateTensors.cend()) {
          frontier.push_back(tensor);
          beenOnFrontier.insert(tid);
        }
      }
    }

    // Starting from the initial frontier found above,
    // propogate "CHECKPOINT" forward til either a Stash Tensor or an IPU copy
    // is reached.
    while (!frontier.empty()) {
      Tensor *tensor = frontier.back();
      frontier.pop_back();
      for (Op *consumer : tensor->consumers.getOps()) {
        consumer->settings.recomputeType = RecomputeType::CHECKPOINT;
        if (!dynamic_cast<IpuCopyOp *>(consumer)) {
          for (Tensor *consumerOut : consumer->output->tensors()) {
            if (beenOnFrontier.count(consumerOut->id) == 0 &&
                // consumerOut is not a stash candidate
                (std::find(toStashCandidateTensors.cbegin(),
                           toStashCandidateTensors.cend(),
                           consumerOut->id) ==
                 toStashCandidateTensors.cend())) {
              frontier.push_back(consumerOut);
              beenOnFrontier.insert(consumerOut->id);
            }
          }
        }
      }
    }

    // Filter stash candidates: only stash CHECKPOINT Ops
    for (auto tid : toStashCandidateTensors) {
      auto tensor = graph.getTensors().get(tid);
      if (!tensor->hasProducer() ||
          tensor->getProducer()->settings.recomputeType !=
              RecomputeType::RECOMPUTE) {
        toStashTensors.push_back(tid);
      }
    }
  }

  // 2. For each Tensor to be stashed, create a single stash
  //    and (in-place) restore op
  Op::Settings settings(graph, "");

  for (auto &tid : toStashTensors) {
    auto tensor               = graph.getTensors().get(tid);
    auto tidConsumers         = tensor->consumers.getOps();
    auto pipelineStageCheckOp = tidConsumers[0];

    // Stash
    auto stashOp_up =
        std::make_unique<StashOp>(Onnx::CustomOperators::Stash, settings);
    auto stashOp = stashOp_up.get();
    graph.moveIntoGraph(std::move(stashOp_up));
    stashOp->setVirtualGraphId(
        getVirtualGraphIdOrSourceIpu(pipelineStageCheckOp));
    stashOp->setPipelineStage(pipelineStageCheckOp->getPipelineStage());
    stashOp->connectInTensor(StashOp::getInIndex(), tid);
    auto stashId = stashOp->getStashedTensorId();
    stashOp->createAndConnectOutTensor(StashOp::getOutIndex(), stashId);
    stashOp->setup();

    logging::transform::debug(
        "Adding stash of size {} of activations {} for pipelining",
        stashOp->getStashSize(),
        tensor->id);

    // Restore

    // Should op be Restore (outplace) or RestoreInplace?
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

    // RECOMPUTE ops must be inplace, confirm:
    for (Op *tidConsumer : tidConsumers) {
      if (tidConsumer->settings.recomputeType == RecomputeType::RECOMPUTE) {
        if (isInplace == false) {
          throw error("A recompute Op consumes a stashed Tensor, therefore "
                      "the stashing must be in-place. But some previous logic "
                      "has set the stashing to be non-inplace");
        }
      }
    }

    RestoreOp *restoreOp;
    if (isInplace) {
      restoreOp = addNewRestoreInplaceOp(graph);
    } else {
      restoreOp = addNewRestoreOp(graph);
    }

    restoreOp->setVirtualGraphId(
        getVirtualGraphIdOrSourceIpu(pipelineStageCheckOp));
    restoreOp->setPipelineStage(pipelineStageCheckOp->getPipelineStage());
    restoreOp->connectInTensor(RestoreOp::getActToRestoreInIndex(), tid);
    restoreOp->connectInTensor(RestoreOp::getStashInIndex(), stashId);
    auto restoreId = restoreOp->getRestoredTensorId();
    restoreOp->createAndConnectOutTensor(RestoreOp::getRestoredActOutIndex(),
                                         restoreId);

    // Disconnect tid from all post-other consumers, reconnect to restoreId
    for (Op *tidConsumer : tidConsumers) {
      if (tidConsumer->scheduledPreLoss == ScheduledPreLoss::No) {
        for (auto i : tidConsumer->input->indicesMap().at(tensor)) {
          tidConsumer->disconnectInTensor(i, tensor);
          tidConsumer->connectInTensor(i, restoreId);
        }
      }
    }

    restoreOp->setup();

    // apply topological constraints:
    // (0)  : Stash before all other consumers
    // (1)  : Stash -> "firstFromLoss" -> Restore
    // The only topological constraint on Restore is that it is
    // SchedulePreLoss::No, exact scheduling controlled by the backend.

    // (1)
    graph.topoCons->insert(stashOp, *firstFromLoss);
    graph.topoCons->insert(*firstFromLoss, restoreOp);

    for (auto tidConsumer : tidConsumers) {

      // (0)
      if (tidConsumer != stashOp) {
        graph.topoCons->insert(stashOp, tidConsumer);
      }
    }
  }
  return true;
}

RestoreOp *Pipeline::addNewRestoreOp(Graph &graph) const {
  Op::Settings settings(graph, "");
  auto restoreOp_up =
      std::make_unique<RestoreOp>(Onnx::CustomOperators::Restore, settings);
  auto restoreOp = restoreOp_up.get();
  graph.moveIntoGraph(std::move(restoreOp_up));

  return restoreOp;
}

RestoreInplaceOp *Pipeline::addNewRestoreInplaceOp(Graph &graph) const {
  Op::Settings settings(graph, "");
  auto restoreOp_up = std::make_unique<RestoreInplaceOp>(
      Onnx::CustomOperators::RestoreInplace, settings);
  auto restoreOp = restoreOp_up.get();
  graph.moveIntoGraph(std::move(restoreOp_up));

  return restoreOp;
}

namespace {
bool init = Transform::registerTransform(new Pipeline);
}

} // namespace popart
