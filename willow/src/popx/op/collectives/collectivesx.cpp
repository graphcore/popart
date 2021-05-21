// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poputil/Util.hpp>

namespace popart {
namespace popx {

popops::CollectiveOperator getPoplarCollectiveOperator(CollectiveOperator op) {
  switch (op) {
  case CollectiveOperator::Add:
    return popops::CollectiveOperator::ADD;
  case CollectiveOperator::Mul:
    return popops::CollectiveOperator::MUL;
  case CollectiveOperator::Min:
    return popops::CollectiveOperator::MIN;
  case CollectiveOperator::Max:
    return popops::CollectiveOperator::MAX;
  case CollectiveOperator::LogicalAnd:
    return popops::CollectiveOperator::LOGICAL_AND;
  case CollectiveOperator::LogicalOr:
    return popops::CollectiveOperator::LOGICAL_OR;
  case CollectiveOperator::SquareAdd:
    return popops::CollectiveOperator::SQUARE_ADD;
  case CollectiveOperator::Local:
    return popops::CollectiveOperator::LOCAL;
  default:
    throw error("Unsupported operator {}", static_cast<int>(op));
  }
}

CollectivesBaseOpx::CollectivesBaseOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {}

std::pair<std::set<TensorId>, std::vector<Op *>>
CollectivesBaseOpx::getCollectiveLinkedGroup() const {

  std::set<TensorId> groupTensorIds;
  std::vector<Op *> groupCollectiveOps;

  Shape shape;
  Shape metaShape;
  if (op_p->isConvertibleTo<ReplicatedReduceScatterOp>()) {
    shape =
        op_p->outTensor(ReplicatedReduceScatterOp::getOutIndex())->info.shape();
    metaShape = op_p->outTensor(ReplicatedReduceScatterOp::getOutIndex())
                    ->info.metaShape();
  }
  if (op_p->isConvertibleTo<ReplicatedAllGatherOp>()) {
    shape = op_p->inTensor(ReplicatedAllGatherOp::getInIndex())->info.shape();
    metaShape =
        op_p->inTensor(ReplicatedAllGatherOp::getInIndex())->info.metaShape();
  }

  auto visitor =
      [&shape, &metaShape, &groupTensorIds, &groupCollectiveOps](Tensor *t) {
        bool keep_going = false;

        for (Op *c : t->consumers.getOps()) {
          if (CollectivesBaseOp *collectiveOp =
                  dynamic_cast<CollectivesBaseOp *>(c)) {
            auto indices = collectiveOp->input->indices(t);
            if (std::find(indices.begin(),
                          indices.end(),
                          CollectivesBaseOp::getCollectiveLinkedIndex()) !=
                indices.end()) {
              for (auto root : graphutils::rootTensors(t)) {
                groupTensorIds.insert(root->id);
              }
              groupCollectiveOps.push_back(collectiveOp);
              keep_going = true;
            }
          }
        }

        if (t->isRemoteArgTensor()) {
          keep_going = true;
        }

        // Same meta shape -> connected RTS domain
        if (t->info.metaShape() == metaShape) {
          keep_going = true;
        }

        if (t->info.shape() == shape && t->info.metaShape() != metaShape) {
          logging::opx::warn("[CollectivesBaseOpx::getCollectiveLinkedGroup] "
                             "tensor {} matches in shape ({} vs. {}) but not "
                             "meta-shape ({} vs. {})",
                             t->id,
                             t->info.shape(),
                             shape,
                             t->info.metaShape(),
                             metaShape);
        }

        logging::opx::trace("[CollectivesBaseOpx::getCollectiveLinkedGroup] "
                            "visiting: {} (keep_going: {})",
                            t->id,
                            keep_going ? "true" : "false");

        return keep_going;
      };

  auto filter = [](Op *op, Tensor *tq, Tensor *tn) {
    // Subgraph inputs/outputs should be traversed
    if (op->isConvertibleTo<SubgraphOp>()) {
      return true;
    }

    // Collective ops should be traversed
    if (op->isConvertibleTo<CollectivesBaseOp>()) {
      return true;
    }

    // All other ops should be traversed if the input/output tensors
    // are RTS related
    auto rtsIndices = op->getReplicatedTensorShardingIndices();

    std::vector<InIndex> tqIn;
    std::vector<InIndex> tnIn;
    std::vector<OutIndex> tqOut;
    std::vector<OutIndex> tnOut;

    if (op->input->contains(tq)) {
      tqIn = op->input->indices(tq);
    }
    if (op->input->contains(tn)) {
      tnIn = op->input->indices(tn);
    }
    if (op->output->contains(tq)) {
      tqOut = op->output->indices(tq);
    }
    if (op->output->contains(tn)) {
      tnOut = op->output->indices(tn);
    }

    for (auto rtsIndex : rtsIndices) {
      bool tqInSet = false;
      bool tnInSet = false;
      for (auto index : tqIn) {
        tqInSet |= rtsIndex.first.find(index) != rtsIndex.first.end();
      }
      for (auto index : tqOut) {
        tqInSet |= rtsIndex.second.find(index) != rtsIndex.second.end();
      }
      for (auto index : tnIn) {
        tnInSet |= rtsIndex.first.find(index) != rtsIndex.first.end();
      }
      for (auto index : tnOut) {
        tnInSet |= rtsIndex.second.find(index) != rtsIndex.second.end();
      }
      if (tqInSet && tnInSet) {
        // Input and output tensor are in the same RTS domain
        return true;
      }
    }

    return false;
  };

  graphutils::traverse(
      {op_p->inTensor(CollectivesBaseOp::getCollectiveLinkedIndex())},
      visitor,
      filter,
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::ForwardBackward);

  const liveness::LivenessAnalyzer *liveness =
      dv_p->lowering().getLivenessAnalyzer();

  // Sort by schedule order in the IR
  std::sort(groupCollectiveOps.begin(),
            groupCollectiveOps.end(),
            [liveness](Op *lhs, Op *rhs) {
              return liveness->getScheduleIndices(lhs).front() <
                     liveness->getScheduleIndices(rhs).front();
            });

  logging::opx::trace(
      "[CollectivesBaseOpx::getCollectiveLinkedGroup] group: {}",
      groupTensorIds);
  return {groupTensorIds, groupCollectiveOps};
}

gcl::CollectiveBalancedReorder *
CollectivesBaseOpx::getCollectiveBalancedReorder() const {
  auto group = getCollectiveLinkedGroup();
  logging::opx::trace("[CollectivesBaseOpx] Getting CBR for {}",
                      *group.first.begin());
  auto cbr =
      dv_p->lowering().getCollectiveBalancedReorder(*group.first.begin());
  return cbr.get();
}

gcl::CollectiveBalancedReorder *
CollectivesBaseOpx::createCollectiveBalancedReorder(
    poplar::Tensor tensor) const {
  auto replicationFactor = dv_p->lowering().getReplicationFactor();
  auto group             = getCollectiveLinkedGroup();
  auto cbr =
      dv_p->lowering().getCollectiveBalancedReorder(*group.first.begin());
  if (!cbr.get()) {
    cbr = std::make_shared<gcl::CollectiveBalancedReorder>(
        graph().getPoplarGraph(),
        tensor,
        replicationFactor,
        getDebugNameAndId());
    for (auto tensor_id : group.first) {
      logging::opx::trace("[CollectivesBaseOpx] CBR created for {}", tensor_id);
      dv_p->lowering().setCollectiveBalancedReorder(tensor_id, cbr);
    }
  }
  return cbr.get();
}

} // namespace popx
} // namespace popart
