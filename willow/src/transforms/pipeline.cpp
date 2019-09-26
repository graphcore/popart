#include <vector>

#include <boost/range/algorithm.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/patterns/contiguateipucopyindices.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/pipeline.hpp>
#include <popart/vertex.hpp>

using boost::range::max_element;
using boost::range::min_element;

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
  ss << "ILE: No candidates for restore op.";

  ss << fmt::format("\nTensor: {}", t->id);
  if (t->hasProducer()) {
    auto prod = t->getProducer();
    ss << fmt::format("\n  Producer: {}, ps: {}, vg: {}",
                      prod->debugName(),
                      prod->getPipelineStage(),
                      getVirtualGraphIdOrSourceIpu(prod));
  }
  ss << "\n  Consumers:";
  for (auto c : t->consumers.getOps()) {
    ss << fmt::format("\n    {}, ps: {}, vg: {}",
                      c->debugName(),
                      c->getPipelineStage(),
                      getVirtualGraphIdOrSourceIpu(c));
  }

  ss << fmt::format("\nStash Ref Op: {}, ps: {}, vg: {}",
                    stashRefOp->debugName(),
                    stashRefOp->getPipelineStage(),
                    getVirtualGraphIdOrSourceIpu(stashRefOp));

  return ss.str();
}

Op *searchForRestoreReferenceOp(Tensor *t) {
  // Find a restore reference Op in the Post Loss graph by searching through
  // the consumers but not crossing IPU boundaries.
  std::vector<Op *> frontier;
  std::set<TensorId> beenOnFrontier = {t->id};
  for (auto *c : t->consumers.getOps()) {
    frontier.push_back(c);
  }
  while (!frontier.empty()) {
    Op *op = frontier.back();
    frontier.pop_back();
    if (!op->isIpuCopyOp()) {
      // If it's post loss return it.
      if (op->scheduledPreLoss == ScheduledPreLoss::No) {
        return op;
      } else {
        // Otherwise go to the output's consumers and add recompute ops to the
        // frontier
        for (Tensor *outT : op->output->tensors()) {
          if (beenOnFrontier.count(outT->id) == 0) {
            beenOnFrontier.insert(outT->id);
            for (auto *c : outT->consumers.getOps()) {
              frontier.push_back(c);
            }
          }
        }
      }
    }
  }
  return nullptr;
}

bool isStashCandidateForPreLossOnly(Tensor *tensor) {
  if (!tensor->consumersAllPreLoss()) {
    return false;
  }
  // If a Tensor is only consumed by ipuCopies then it shouldn't be stashed
  for (auto *c : tensor->consumers.getOps()) {
    if (!c->isIpuCopyOp()) {
      return true;
    }
  }
  return false;
}

Op *getRestoreReferenceOp(Tensor *t, Op *stashRefOp) {
  logging::debug("Collecting restore ref candidates");
  auto consumers = t->consumers.getOps();

  std::vector<Op *> restoreCandidates;
  if (t->consumersAllPreLoss()) {
    auto *ref = searchForRestoreReferenceOp(t);
    if (ref != nullptr)
      restoreCandidates.push_back(ref);
  } else {
    for (auto c : consumers) {
      if (getVirtualGraphIdOrSourceIpu(c) ==
              getVirtualGraphIdOrSourceIpu(stashRefOp) &&
          c->getPipelineStage() != stashRefOp->getPipelineStage()) {
        restoreCandidates.push_back(c);
      }
    }
  }

  if (restoreCandidates.size() == 0) {
    throw error(zeroCandidatesError(t, stashRefOp));
  }

  // Check all candidates have the same pipeline stage
  PipelineStage restorePipelineStage =
      restoreCandidates.at(0)->getPipelineStage();
  for (auto c : restoreCandidates) {
    if (restorePipelineStage != c->getPipelineStage()) {
      throw error("Conflicting candidates for restore op pipeline stage");
    }
  }

  return restoreCandidates.at(0);
}

} // namespace

bool Pipeline::apply(Graph &graph) const {

  auto &ir         = graph.getIr();
  auto maxVGraphId = ir.getMaxVirtualGraphId();
  bool full_recompute =
      ir.getSessionOptions().autoRecomputation == RecomputationType::Pipeline;
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
    std::set<VGraphId> virtualGraphs;
    for (auto c : tensor->consumers.getOps()) {
      virtualGraphs.insert(getVirtualGraphIdOrSourceIpu(c));
    }

    if (virtualGraphs.size() > 1) {
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

  // 1. Find all tensors in the fwd pass that are inputs to ops in the bwd pass
  std::vector<TensorId> toStashCandidateTensors;
  for (auto &tid : graph.getTensors().getAllTensorIds()) {
    auto tensor = graph.getTensors().get(tid);

    // Not a candidate for stashing if the tensor:
    // - has no consumers
    // - is a variable tensor
    // - is an optimizer tensor
    // - is a constant tensor

    if (tensor->consumers.getOps().empty()) {
      continue;
    }
    if (tensor->tensorType() == TensorType::Variable) {
      continue;
    }
    if (tensor->tensorType() == TensorType::Const) {
      continue;
    }
    if (tensor->isOptimizerTensor()) {
      continue;
    }

    // Get all the stages the tensor is produced/consumed in.
    std::set<PipelineStage> tensorStages = tensor->getPipelineStages();

    // There is no need to stash a tensor that only appears in 1 stage.
    if (tensorStages.size() == 1 &&
        !(full_recompute && isStashCandidateForPreLossOnly(tensor))) {
      continue;
    }

    logging::transform::debug("Adding {} to stash candidates", tid);
    toStashCandidateTensors.push_back(tid);
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
        // If using full_recompute all forward ops are Recomputed
        if (!full_recompute) {
          for (auto tensor : op->output->tensors()) {
            for (auto consumer : tensor->consumers.getOps()) {
              if (dynamic_cast<IpuCopyOp *>(consumer)) {
                op->settings.recomputeType = RecomputeType::CHECKPOINT;
              }
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
    auto tensor = graph.getTensors().get(tid);

    auto stashRefOp   = getStashReferenceOp(tensor);
    auto restoreRefOp = getRestoreReferenceOp(tensor, stashRefOp);

    auto stashSize =
        restoreRefOp->getPipelineStage() - stashRefOp->getPipelineStage() + 1;

    // Stash
    auto stashOp_up = std::make_unique<StashOp>(
        Onnx::CustomOperators::Stash, stashSize, settings);
    auto stashOp = stashOp_up.get();
    graph.moveIntoGraph(std::move(stashOp_up));
    stashOp->setVirtualGraphId(getVirtualGraphIdOrSourceIpu(stashRefOp));
    stashOp->setPipelineStage(stashRefOp->getPipelineStage());
    stashOp->connectInTensor(StashOp::getInIndex(), tid);
    auto stashId = stashOp->getStashedTensorId();
    stashOp->createAndConnectOutTensor(StashOp::getOutIndex(), stashId);
    stashOp->setup();

    logging::transform::debug("Adding stash of size {} of activations {} for "
                              "pipelining. Stash stage: {}, Restore stage {}",
                              stashOp->getStashSize(),
                              tensor->id,
                              stashOp->getPipelineStage(),
                              restoreRefOp->getPipelineStage());

    // Full Recomputation
    // If one of the preLossOnly stash tensors is consumed by an IpuCopy
    // it must not be inplace, but stashes needed for recomputation must be
    // inplace. To resolve this contradiction an IdentityOp is inserted between
    // the the stashed tensor and the IpuCopy
    if (full_recompute && tensor->consumersAllPreLoss()) {
      std::vector<IpuCopyOp *> ipuCopyConsumers;
      for (auto *c : tensor->consumers.getOps()) {
        auto ipuCopyOp = dynamic_cast<IpuCopyOp *>(c);
        if (ipuCopyOp)
          ipuCopyConsumers.push_back(ipuCopyOp);
      }
      if (ipuCopyConsumers.size() > 0) {
        logging::transform::debug("Adding Identity Copy for stash tensor {}",
                                  tid);
        Op::Settings identitySettings(graph, tid + "_pipelineCopyOp");
        TensorId identityOutput = tid + "_pipelineCopy";

        // TODO: Make sure this is not pruned or inplaced. T11668
        auto op = std::make_unique<IdentityOp>(Onnx::Operators::Identity_1,
                                               identitySettings);

        if (op == nullptr) {
          throw error("Failed to create op {}", Onnx::Operators::Identity_1);
        }

        auto src_ipu = getVirtualGraphIdOrSourceIpu(stashRefOp);
        op->connectInTensor(0, tid);
        op->createAndConnectOutTensor(0, identityOutput);
        op->setVirtualGraphId(src_ipu);
        op->setPipelineStage(stashRefOp->getPipelineStage());

        op->setup();
        graph.moveIntoGraph(std::move(op));
        for (auto *ipuCopyOp : ipuCopyConsumers) {
          InIndex index = -1;
          for (auto input : ipuCopyOp->input->tensorIdMap()) {
            if (input.second == tid) {
              index = input.first;
              break;
            }
          }
          if (index == -1) {
            throw error("Could not determine input index for {}", tid);
          }
          ipuCopyOp->disconnectInTensor(index, tensor);
          ipuCopyOp->connectInTensor(index, identityOutput, src_ipu);
        }
      }
    }

    // Restore
    auto tidConsumers = tensor->consumers.getOps();

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
      restoreOp = addNewRestoreInplaceOp(graph, stashSize);
    } else {
      restoreOp = addNewRestoreOp(graph, stashSize);
    }

    restoreOp->setVirtualGraphId(getVirtualGraphIdOrSourceIpu(restoreRefOp));
    restoreOp->setPipelineStage(restoreRefOp->getPipelineStage());
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

    // StashOp should be before all other consumers
    for (auto tidConsumer : tidConsumers) {
      if (tidConsumer != stashOp) {
        graph.topoCons->insert(stashOp, tidConsumer);
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

namespace {
bool init = Transform::registerTransform(new Pipeline);
}

} // namespace popart
