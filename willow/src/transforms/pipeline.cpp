#include <poponnx/error.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/ipucopy.hpp>
#include <poponnx/op/restore.hpp>
#include <poponnx/op/stash.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>

#include <poponnx/transforms/pipeline.hpp>

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
//  t_act ------             t_act_grad
//   |          |               |
//   |        StashOp           |
//   |          |               |
//   |        t_act_stashed     |
//   |          |               |
// RestoreOp <--                |
//   |                          |
// t_act_alias --------------- BwdOp
//   |                          |
//  ...                     t_grad_in

namespace poponnx {

std::size_t Pipeline::id() { return typeid(Pipeline).hash_code(); }

bool Pipeline::apply(Graph &graph) const {
  auto &ir     = graph.getIr();
  auto numIPUs = ir.getDeviceInfo()->getNumIpus();

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
  if (ir.getDataFlow().batchesPerStep() < minDepth) {
    throw error("For pipelining, depth must be at least " +
                std::to_string(minDepth) + " for " +
                std::to_string(ir.getDeviceInfo()->getNumIpus()) + " IPUs");
  }

  // 3. Currently recomputation is not supported with pipelining (TODO T9575)
  if (ir.autoRecomputationEnabled() ||
      ir.getMainGraph().hasUserRecomputeOps()) {
    throw error(
        "When pipelining is enabled, recomputation is currently not allowed");
  }

  // 4. We assume layers must be sharded linearly with IPU number.
  //    Examples violating this:
  //      - Cases where an op on IPU N cannot depends on an op on IPU n>N
  //          Consider the fwd Graph : Op0 -> Op1 -> Op2 -> Op3
  //          e.g. 1) IPU0 : {Op2, Op3}, IPU1 : {Op0, Op1}
  //          e.g. 2) IPU0 : {Op0, Op2}, IPU1 : {Op1, Op3}
  //      - Parallel branches, such as Op0 -> Op2
  //                                           ^
  //                                   Op1 ----'
  //        where the vGraph split is IPU0 : {Op0}, IPU1 : {Op1}, IPU2 : {Op2}.

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

    // 4.2 All copy ops with pathToLoss=yes go from IPU N->N+1,
    //     and all copy ops with pathFromLoss=yes go from IPU N->N-1
    if (op->isIpuCopyOp()) {
      auto ipuCopyOp = dynamic_cast<poponnx::IpuCopyOp *>(op);
      auto sourceIpu = ipuCopyOp->getSourceIpu();
      auto destIpu   = ipuCopyOp->getDestIpu();

      // For an inference graph, or fwd pass of a training graph...
      if (op->toLoss == PathToLoss::Yes || !ir.canTrain()) {
        if (destIpu != sourceIpu + 1) {
          throw error("For pipelining, the graph must be sharded such that "
                      "forward IPU copies go from IPU N to N+1. However, " +
                      op->debugName() + " copies from " +
                      std::to_string(sourceIpu) + " to " +
                      std::to_string(destIpu));
        }
      }
      // For the bwd pass of a training graph...
      else {
        if (destIpu != sourceIpu - 1) {
          throw error("For pipelining, the graph must be sharded such that "
                      "backward IPU copies go from IPU N to N-1. However, " +
                      op->debugName() + " copies from " +
                      std::to_string(sourceIpu) + " to " +
                      std::to_string(destIpu));
        }
      }
    }
  }

  // Other sharding assumptions to check:

  // 5. Ir stream tensors cannot be consumed by ops on multiple IPUs
  for (TensorId tid : graph.getTensors().getIds(TensorType::Stream)) {
    auto tensor                = graph.getTensors().get(tid);
    auto consumerOps           = tensor->consumers.getOps();
    auto firstConsumerVGraphId = consumerOps[0]->getVirtualGraphId();
    for (Op *consumer : consumerOps) {
      if (consumer->getVirtualGraphId() != firstConsumerVGraphId) {
        throw error("For pipelining, stream tensors can only be streamed "
                    "directly onto a single IPU");
      }
    }
  }

  // Now apply the transform

  if (!ir.canTrain()) {
    // No stashing of forward activations required in inference/eval mode
    return true;
  }

  // 1. Find all tensors in the fwd pass that are inputs to ops in the bwd pass
  std::vector<TensorId> toStashTensors;
  for (auto &tid : graph.getTensors().getAllTensorIds()) {
    auto tensor = graph.getTensors().get(tid);

    // Not a candidate for stashing if the tensor:
    // - has no consumers
    // - is a variable tensor
    // - is on the final IPU
    if (tensor->consumers.getOps().empty()) {
      continue;
    }
    if (tensor->tensorType() == TensorType::Variable) {
      continue;
    }

    auto vGraphIdCheckOp = tensor->consumers.getOps()[0];
    int vGraphId = static_cast<int>(vGraphIdCheckOp->getVirtualGraphId());
    if (vGraphId == numIPUs - 1) {
      continue;
    }

    bool isConsumedByFwdOp = false;
    bool isConsumedByBwdOp = false;
    for (Op *consumer : tensor->consumers.getOps()) {
      if (consumer->toLoss == PathToLoss::Yes) {
        isConsumedByFwdOp = true;
      } else if (consumer->scheduledPreLoss == ScheduledPreLoss::No) {
        isConsumedByBwdOp = true;
      }
    }

    if (isConsumedByFwdOp && isConsumedByBwdOp) {
      toStashTensors.push_back(tid);
    }
  }

  // 2. For each acivation tensor, create a single stash
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
    stashOp->setVirtualGraphId(vGraphIdCheckOp->getVirtualGraphId());
    stashOp->connectInTensor(StashOp::getInIndex(), tid);
    auto stashId = stashOp->getStashedTensorId();
    stashOp->createAndConnectOutTensor(StashOp::getOutIndex(), stashId);
    stashOp->setup();

    // Restore
    auto restoreOp_up =
        std::make_unique<RestoreOp>(Onnx::CustomOperators::Restore, settings);
    auto restoreOp = restoreOp_up.get();
    graph.moveIntoGraph(std::move(restoreOp_up));
    restoreOp->setVirtualGraphId(vGraphIdCheckOp->getVirtualGraphId());
    restoreOp->connectInTensor(RestoreOp::getActToRestoreInIndex(), tid);
    restoreOp->connectInTensor(RestoreOp::getStashInIndex(), stashId);
    auto restoreId = restoreOp->getRestoredTensorId(); // An alias
    restoreOp->createAndConnectOutTensor(RestoreOp::getRestoredActOutIndex(),
                                         restoreId);
    // Disconnect tid from all other consumers, reconnect to restoreId
    for (Op *tidConsumer : tidConsumers) {
      for (auto i : tidConsumer->input->indicesMap().at(tensor)) {
        tidConsumer->disconnectInTensor(i, tensor);
        tidConsumer->connectInTensor(i, restoreId);
      }
    }
    restoreOp->setup();

    logging::transform::debug(
        "Adding stash of size {} of activations {} for pipelining",
        stashOp->getStashSize(),
        tensor->id);
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new Pipeline);
}

} // namespace poponnx
