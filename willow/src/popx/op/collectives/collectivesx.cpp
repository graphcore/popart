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
#include <popart/popx/replicatedtensorshardingbundle.hpp>
#include <popart/replicatedtensorsharding.hpp>

#include <poputil/Util.hpp>

namespace popart {
namespace popx {

popops::CollectiveOperator getPoplarCollectiveOperator(CollectiveOperator op) {
  switch (op) {
  case CollectiveOperator::Add:
    return popops::CollectiveOperator::ADD;
  case CollectiveOperator::Mean:
    return popops::CollectiveOperator::MEAN;
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

ReplicatedTensorShardingGroup
CollectivesBaseOpx::getCollectiveLinkedGroup() const {

  auto &tracer = dv_p->lowering()
                     .getReplicatedTensorShardingBundle()
                     .getReplicatedTensorShardingTracer();

  auto rtsIndices = op_p->getReplicatedTensorShardingIndices();

  ReplicatedTensorShardingOpInfo id{
      op_p->id, rtsIndices.begin()->first, rtsIndices.begin()->second};

  if (!tracer.hasGroup(id)) {
    bool hasLinkedInput =
        op_p->hasInput(CollectivesBaseOp::getCollectiveLinkedIndex());

    std::set<Tensor *, PTensorCmp> startTensors;

    if (hasLinkedInput) {
      startTensors.insert(
          op_p->inTensor(CollectivesBaseOp::getCollectiveLinkedIndex()));
    }

    if (op_p->isConvertibleTo<ReplicatedReduceScatterOp>()) {
      startTensors.insert(
          op_p->outTensor(ReplicatedReduceScatterOp::getOutIndex()));
    }

    if (op_p->isConvertibleTo<ReplicatedAllGatherOp>()) {
      startTensors.insert(op_p->inTensor(ReplicatedAllGatherOp::getInIndex()));
    }

    tracer.trace(startTensors);
  }
  auto group = tracer.getGroup(id);

  logging::opx::trace(
      "[CollectivesBaseOpx::getCollectiveLinkedGroup] group: {}", group);
  return group;
}

gcl::CollectiveBalancedReorder *
CollectivesBaseOpx::getCollectiveBalancedReorder() const {
  auto group = getCollectiveLinkedGroup();

  TensorId tensorIdForCBR;

  if (!group.collectiveLinkedTensorIds.empty()) {
    tensorIdForCBR = *group.collectiveLinkedTensorIds.begin();
  } else if (!group.shardedTensorIds.empty()) {
    tensorIdForCBR = *group.shardedTensorIds.begin();
  } else {
    throw error("[CollectivesBaseOpx::getCollectiveBalancedReorder] Could not "
                "find replicated tensor sharding group for {}",
                op_p->debugName());
  }

  logging::opx::trace("[CollectivesBaseOpx] Getting CBR for {}",
                      tensorIdForCBR);
  auto cbr = dv_p->lowering()
                 .getReplicatedTensorShardingBundle()
                 .getCollectiveBalancedReorder(tensorIdForCBR);
  return cbr.get();
}

gcl::CollectiveBalancedReorder *
CollectivesBaseOpx::createCollectiveBalancedReorder(snap::Tensor tensor) const {
  auto globalReplicationFactor = dv_p->lowering().getGlobalReplicationFactor();
  auto replicationFactor       = globalReplicationFactor;
  auto group                   = getCollectiveLinkedGroup();

  TensorId tensorIdForCBR;

  if (!group.collectiveLinkedTensorIds.empty()) {
    tensorIdForCBR = *group.collectiveLinkedTensorIds.begin();
  } else if (!group.shardedTensorIds.empty()) {
    tensorIdForCBR = *group.shardedTensorIds.begin();
  } else {
    throw error(
        "[CollectivesBaseOpx::createCollectiveBalancedReorder] Could not "
        "find replicated tensor sharding group for {}",
        op_p->debugName());
  }

  for (auto opId : group.collectiveOpIds) {
    if (auto collective =
            dynamic_cast<CollectivesBaseOp *>(dv_p->ir().getOp(opId.first))) {
      auto shardingDomain = collective->getGCLCommGroup();
      if (shardingDomain.replicaGroupSize > 0 &&
          (shardingDomain.type == CommGroupType::Consecutive ||
           shardingDomain.type == CommGroupType::Orthogonal)) {
        replicationFactor = shardingDomain.replicaGroupSize;
      }
    }
  }

  gcl::CollectiveBalancedReorder *cbrPtr;
  bool hasCbr = dv_p->lowering()
                    .getReplicatedTensorShardingBundle()
                    .hasCollectiveBalancedReorder(tensorIdForCBR);
  if (hasCbr) {
    auto cbr = dv_p->lowering()
                   .getReplicatedTensorShardingBundle()
                   .getCollectiveBalancedReorder(tensorIdForCBR);
    cbrPtr = cbr.get();
  } else {
    auto cbr = std::make_shared<gcl::CollectiveBalancedReorder>(
        graph().getPoplarGraph(),
        tensor.getPoplarTensor(),
        replicationFactor,
        getDebugNameAndId());
    auto cbrId = dv_p->lowering()
                     .getReplicatedTensorShardingBundle()
                     .registerCollectiveBalancedReorder(cbr);
    cbrPtr = cbr.get();

    auto setCBR = [this, &cbrId, &replicationFactor, &globalReplicationFactor](
                      TensorId &tensorId) {
      logging::opx::trace(
          "[CollectivesBaseOpx] CBR with ID {} created for {}, sharding "
          "factor: {}, global replication factor: {}",
          cbrId,
          tensorId,
          replicationFactor,
          globalReplicationFactor);
      dv_p->lowering()
          .getReplicatedTensorShardingBundle()
          .setCollectiveBalancedReorder(tensorId, cbrId);
    };

    for (auto tensorId : group.remoteTensorIds) {
      setCBR(tensorId);
    }
    for (auto tensorId : group.collectiveLinkedTensorIds) {
      setCBR(tensorId);
    }
    for (auto tensorId : group.shardedTensorIds) {
      setCBR(tensorId);
    }
  }
  return cbrPtr;
}

gcl::CommGroup toGCLCommGroup(const popart::CommGroup &group) {
  gcl::CommGroupType type;
  switch (group.type) {
  case popart::CommGroupType::All:
    type = gcl::CommGroupType::ALL;
    break;
  case popart::CommGroupType::Consecutive:
    type = gcl::CommGroupType::CONSECUTIVE;
    break;
  case popart::CommGroupType::Orthogonal:
    type = gcl::CommGroupType::ORTHOGONAL;
    break;
  default:
    throw error("Cannot convert unknown CommGroup type");
  }
  return {type, group.replicaGroupSize};
}

} // namespace popx
} // namespace popart
