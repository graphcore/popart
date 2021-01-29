// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op/collectives/collectives.hpp>
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
    : Opx(op, devicex) {}

std::pair<std::set<TensorId>, std::vector<Op *>>
CollectivesBaseOpx::getCollectiveLinkedGroup() const {
  // Mapping from each RemoteArg to it's final consumers
  std::map<TensorId, std::set<Op *>> linkOpMap;
  std::map<Op *, std::set<TensorId>> opLinkMap;

  Ir &ir = op_p->getIr();
  const liveness::LivenessAnalyzer *liveness =
      dv_p->lowering().getLivenessAnalyzer();

  for (Op *op : ir.getAllOps()) {
    if (CollectivesBaseOp *collectiveOp =
            dynamic_cast<CollectivesBaseOp *>(op)) {

      if (!collectiveOp->input->hasIndex(
              CollectivesBaseOp::getCollectiveLinkedIndex())) {
        continue;
      }

      std::vector<Tensor *> traceFront;
      traceFront.push_back(collectiveOp->input->tensor(
          CollectivesBaseOp::getCollectiveLinkedIndex()));

      while (traceFront.size() > 0) {
        Tensor *front = traceFront.back();
        traceFront.pop_back();
        auto inputIds = front->getGraph().getInputIds();
        if (front->hasProducer()) {
          // The link tensor is only allowed to come through subgraph ops, and
          // should not be touched by other ops
          throw error("Op {} not expected on the path from the "
                      "link tensor to the collective operation {}",
                      front->getProducer()->debugName(),
                      op->debugName());
        } else {
          auto it = std::find(inputIds.begin(), inputIds.end(), front->id);
          if (it != inputIds.end()) {
            InIndex index =
                static_cast<InIndex>(std::distance(inputIds.begin(), it));
            auto &callSites = liveness->getGraphCallSites(front->getGraph().id);
            for (Op *callSite : callSites) {
              traceFront.push_back(callSite->input->tensor(index));
            }
          } else {
            linkOpMap[front->id].insert(op);
            opLinkMap[op].insert(front->id);
          }
        }
      }
    }
  }

  std::set<TensorId> groupTensorIds;
  std::vector<Op *> groupCollectiveOps;

  std::vector<Op *> front(1, op_p);
  while (front.size() > 0) {
    Op *frontOp = front.back();
    front.pop_back();
    for (TensorId tensor_id : opLinkMap.at(frontOp)) {
      if (groupTensorIds.find(tensor_id) == groupTensorIds.end()) {
        groupTensorIds.insert(tensor_id);
        for (Op *op : linkOpMap.at(tensor_id)) {
          if (std::find(groupCollectiveOps.begin(),
                        groupCollectiveOps.end(),
                        op) == groupCollectiveOps.end()) {
            groupCollectiveOps.push_back(op);
            front.push_back(op);
          }
        }
      }
    }
  }

  // Sort by schedule order in the IR
  std::sort(groupCollectiveOps.begin(),
            groupCollectiveOps.end(),
            [liveness](Op *lhs, Op *rhs) {
              return liveness->getScheduleIndices(lhs).front() <
                     liveness->getScheduleIndices(rhs).front();
            });

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
        graph(), tensor, replicationFactor, getDebugNameAndId());
    for (auto tensor_id : group.first) {
      logging::opx::trace("[CollectivesBaseOpx] CBR created for {}", tensor_id);
      dv_p->lowering().setCollectiveBalancedReorder(tensor_id, cbr);
    }
  }
  return cbr.get();
}

} // namespace popx
} // namespace popart
