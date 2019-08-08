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

bool Pipeline::apply(Graph &graph) const {

  auto &ir     = graph.getIr();
  auto numIPUs = ir.getDeviceInfo()->getNumIpus();
  // We use numIPUs as proxy for sharding factor, not quite correct:TODO T10254

  if (ir.autoRecomputationEnabled()) {
    throw error("Auto recomputation for Pipelining needs implementing");
  }

  // First, some checks that pipelining is compatible with other user options:

  // 1. Pipelining uses the virtual graph API. This must be enabled
  if (!ir.getSessionOptions().enableVirtualGraphs) {
    throw error("Pipelining requires the 'enableVirtualGraphs' session option "
                "to be turned on.");
  }

  // 2. There must be enough batches of data for the cycle of filling
  //    and flushing the pipeline
  int minDepth;
  if (ir.canTrain()) {
    minDepth = 2 * (numIPUs - 1) + 1;
  } else {
    minDepth = numIPUs;
  }

  int depth;
  if (ir.getSessionOptions().enableGradientAccumulation) {
    depth = ir.getSessionOptions().accumulationFactor;
    if (depth < minDepth) {
      // TODO: Update this to account for replication (T10254) . This
      // requirement is more complex when replicating.
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
  for (auto &op_pair : graph.getOps()) {
    Op *op = op_pair.second.get();

    // 4.1. Loss is on graph numIPUs - 1
    if (op->isLossOp()) {
      int vGraphId = static_cast<int>(op->getVirtualGraphId());
      if (vGraphId != numIPUs - 1) {
        throw error("For pipelining, the graph must be sharded such that "
                    "the loss is on the final IPU in the pipeline. "
                    "Loss op " +
                    op->debugName() + " has vGraphId " +
                    std::to_string(vGraphId) + " but there are " +
                    std::to_string(numIPUs) + " IPUs");
      }
    }
  }

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

  auto ipuCopies = getIpuCopyOps();
  for (auto ipuCopyOp : ipuCopies) {
    // Ignore copies of optimizer tensors. These are run inside a different
    // program fragment to the pipelined program, so don't have to follow
    // the same constraints of other IpuCopy ops.
    if (ipuCopyOp->copiesOptimizerTensors()) {
      continue;
    }

    uint64_t destIpu = ipuCopyOp->getDestIpu();
    // For an inference graph, or fwd pass of a training graph,
    if (!ir.canTrain() ||
        ipuCopyOp->scheduledPreLoss == ScheduledPreLoss::Yes) {
      uint64_t maxSourceIpu = ipuCopyOp->getMaxSourceIpu();
      if (destIpu <= maxSourceIpu) {
        throw error(
            "For pipelining, forward IpuCopyOps must copy to an "
            "IPU of larger index. However {} ({}) copies from {} to {}.",
            ipuCopyOp->debugName(),
            ipuCopyOp->str(),
            maxSourceIpu,
            destIpu);
      }
    }

    // for bwd pass, IPU Copy must have destination with lower VGraphId than
    // source
    else if (ipuCopyOp->scheduledPreLoss == ScheduledPreLoss::No) {
      uint64_t minSourceIpu = ipuCopyOp->getMinSourceIpu();
      if (destIpu >= minSourceIpu) {
        throw error(
            "For pipelining, backward IpuCopyOps must copy to an "
            "IPU of smaller index. However {} ({}) copies from {} to {}.",
            ipuCopyOp->debugName(),
            ipuCopyOp->str(),
            minSourceIpu,
            destIpu);
      }
    }
  }

  // Other sharding assumptions to check:

  // 5. Ir stream tensors cannot be consumed by ops on multiple IPUs
  for (TensorId tid : graph.getTensors().getIds(TensorType::Stream)) {
    auto tensor                = graph.getTensors().get(tid);
    auto consumerOps           = tensor->consumers.getOps();
    auto firstConsumerVGraphId = getVirtualGraphIdOrSourceIpu(consumerOps[0]);
    for (Op *consumer : consumerOps) {
      if (getVirtualGraphIdOrSourceIpu(consumer) != firstConsumerVGraphId) {
        throw error("For pipelining, stream tensors can only be streamed "
                    "directly onto a single IPU");
      }
    }
  }

  // Now apply the transform

  // 0. Contiguate the IPUCopies
  ContiguateIpuCopyIndicesPattern contiguator;
  for (auto ipuCopyOp : ipuCopies) {
    if (contiguator.matches(ipuCopyOp)) {
      logging::transform::debug("Contiguating {}", ipuCopyOp->debugName());
      contiguator.apply(ipuCopyOp);
    }
  }
  ir.updateVertices();

  // verify that all IpuCopies are contiguous
  for (auto ipuCopyOp : getIpuCopyOps()) {
    auto sourceIpu = static_cast<int64_t>(ipuCopyOp->getSourceIpu());
    auto destIpu   = static_cast<int64_t>(ipuCopyOp->getDestIpu());
    auto delta     = destIpu - sourceIpu;
    if (delta != 1 && delta != -1) {
      throw error("ILE: IpuCopy {} is not contiguous. It copies from IPU {} to "
                  "IPU {}. Failed to contiguate all IpuCopyOps",
                  ipuCopyOp->debugName(),
                  sourceIpu,
                  destIpu);
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
  std::vector<TensorId> toStashTensors;
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

    auto vGraphIdCheckOp = tensor->consumers.getOps()[0];
    int vGraphId =
        static_cast<int>(getVirtualGraphIdOrSourceIpu(vGraphIdCheckOp));
    if (vGraphId == numIPUs - 1) {
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
      toStashTensors.push_back(tid);
    }
  }

  // 2. For each Tensor to be stashed, create a single stash
  //    and (in-place) restore op
  Op::Settings settings(graph, "");

  for (auto &tid : toStashTensors) {
    auto tensor          = graph.getTensors().get(tid);
    auto tidConsumers    = tensor->consumers.getOps();
    auto vGraphIdCheckOp = tidConsumers[0];

    // Stash
    auto stashOp_up =
        std::make_unique<StashOp>(Onnx::CustomOperators::Stash, settings);
    auto stashOp = stashOp_up.get();
    graph.moveIntoGraph(std::move(stashOp_up));
    stashOp->setVirtualGraphId(getVirtualGraphIdOrSourceIpu(vGraphIdCheckOp));
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

    RestoreOp *restoreOp;
    if (isInplace) {
      restoreOp = addNewRestoreInplaceOp(graph);
    } else {
      restoreOp = addNewRestoreOp(graph);
    }

    restoreOp->setVirtualGraphId(getVirtualGraphIdOrSourceIpu(vGraphIdCheckOp));
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
    // (2)  : All backwards after Restore.

    // (1)
    graph.topoCons->insert(stashOp, *firstFromLoss);
    graph.topoCons->insert(*firstFromLoss, restoreOp);

    for (auto tidConsumer : tidConsumers) {

      // (0)
      if (tidConsumer != stashOp) {
        graph.topoCons->insert(stashOp, tidConsumer);
      }

      // (2)
      // we use "backwards" to mean scheduledPreLoss is No,  but
      // we could equivalently check pathFromLoss is Yes
      if (tidConsumer->scheduledPreLoss == ScheduledPreLoss::No) {
        graph.topoCons->insert(restoreOp, tidConsumer);
      }
    }
  }

  return true;
}

int64_t Pipeline::getVirtualGraphIdOrSourceIpu(Op *op) const {
  if (op->isConvertibleTo<IpuCopyOp>()) {
    auto ipuCopyOp = dynamic_cast<popart::IpuCopyOp *>(op);
    return static_cast<int64_t>(ipuCopyOp->getSourceIpu());
  } else {
    return op->getVirtualGraphId();
  }
}

RestoreOp *Pipeline::addNewRestoreOp(Graph &graph) const {
  Op::Settings settings(graph, "");
  auto restoreOp_up =
      std::make_unique<RestoreOp>(Onnx::CustomOperators::Restore, settings);
  auto restoreOp = restoreOp_up.get();
  graph.moveIntoGraph(std::move(restoreOp_up));

  return restoreOp;
}

RestoreOp *Pipeline::addNewRestoreInplaceOp(Graph &graph) const {
  Op::Settings settings(graph, "");
  auto restoreOp_up = std::make_unique<RestoreOp>(
      Onnx::CustomOperators::RestoreInplace, settings);
  auto restoreOp = restoreOp_up.get();
  graph.moveIntoGraph(std::move(restoreOp_up));

  return restoreOp;
}

namespace {
bool init = Transform::registerTransform(new Pipeline);
}

} // namespace popart
