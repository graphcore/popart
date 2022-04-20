// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <boost/integer/common_factor.hpp>
#include <gcl/CollectiveBalancedReorder.hpp>
#include <gcl/Collectives.hpp>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>
#include <utility>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/replicatedtensorshardingbundle.hpp>
#include <popart/replicatedtensorsharding.hpp>

#include "popart/commgroup.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"

namespace popart {
class Tensor;

namespace popx {

gcl::CollectiveOperator getPoplarCollectiveOperator(CollectiveOperator op) {
  switch (op) {
  case CollectiveOperator::Add:
    return gcl::CollectiveOperator::ADD;
  case CollectiveOperator::Mean:
    return gcl::CollectiveOperator::MEAN;
  case CollectiveOperator::Mul:
    return gcl::CollectiveOperator::MUL;
  case CollectiveOperator::Min:
    return gcl::CollectiveOperator::MIN;
  case CollectiveOperator::Max:
    return gcl::CollectiveOperator::MAX;
  case CollectiveOperator::LogicalAnd:
    return gcl::CollectiveOperator::LOGICAL_AND;
  case CollectiveOperator::LogicalOr:
    return gcl::CollectiveOperator::LOGICAL_OR;
  case CollectiveOperator::SquareAdd:
    return gcl::CollectiveOperator::SQUARE_ADD;
  case CollectiveOperator::Local:
    return gcl::CollectiveOperator::LOCAL;
  default:
    throw error("Unsupported operator {}", static_cast<int>(op));
  }
}

CollectivesBaseOpx::CollectivesBaseOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {}

ReplicatedTensorShardingGroup CollectivesBaseOpx::getCollectiveLinkedGroup(
    ReplicatedTensorShardingIndicesIndex groupIndex) const {
  auto &tracer = dv_p->lowering()
                     .getReplicatedTensorShardingBundle()
                     .getReplicatedTensorShardingTracer();
  ReplicatedTensorShardingIndices rtsIndices =
      op_p->getReplicatedTensorShardingIndices();
  CollectivesBaseOp *collectiveOp = dynamic_cast<CollectivesBaseOp *>(op_p);

  // Return the groupIndex'th group
  ReplicatedTensorShardingIndices::iterator rtsIterator = rtsIndices.begin();
  std::advance(rtsIterator, groupIndex);
  auto p                         = *rtsIterator;
  std::set<InIndex> &inIndices   = p.first;
  std::set<OutIndex> &outIndices = p.second;

  ReplicatedTensorShardingOpInfo id{op_p->id, inIndices, outIndices};
  if (!tracer.hasGroup(id)) {
    std::set<Tensor *, PTensorCmp> startTensors;

    for (InIndex rts_in : inIndices) {
      Tensor *t = collectiveOp->inTensor(rts_in);
      startTensors.insert(t);
      if (collectiveOp->hasCorrespondingLinkedIndexTensor(t)) {
        startTensors.insert(collectiveOp->getCorrespondingLinkedIndexTensor(t));
      }
    }

    for (OutIndex rts_out : outIndices) {
      Tensor *t = collectiveOp->outTensor(rts_out);
      startTensors.insert(t);
      if (collectiveOp->hasCorrespondingLinkedIndexTensor(t)) {
        startTensors.insert(collectiveOp->getCorrespondingLinkedIndexTensor(t));
      }
    }

    tracer.trace(startTensors);
  }

  auto group = tracer.getGroup(id);

  logging::opx::trace(
      "[CollectivesBaseOpx::getCollectiveLinkedGroup] group: {}", group);
  return group;
}

gcl::CollectiveBalancedReorder *
CollectivesBaseOpx::getCollectiveBalancedReorder(
    ReplicatedTensorShardingIndicesIndex groupIndex) const {
  auto group = getCollectiveLinkedGroup(groupIndex);

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

namespace {
unsigned numGrainElements(const poplar::Target &target,
                          const poplar::Type &type) {
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();
  const auto typeSize              = target.getTypeSize(type);
  const auto vecWidth              = target.getVectorWidth(type);
  return boost::integer::lcm<unsigned>(exchangeBytesPerCycle,
                                       vecWidth * typeSize) /
         typeSize;
}
} // namespace

gcl::CollectiveBalancedReorder *
CollectivesBaseOpx::createCollectiveBalancedReorder(
    snap::Tensor tensor,
    ReplicatedTensorShardingIndicesIndex groupIndex) const {
  auto globalReplicationFactor = dv_p->lowering().getGlobalReplicationFactor();
  auto replicationFactor       = globalReplicationFactor;
  auto group                   = getCollectiveLinkedGroup(groupIndex);

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
      } else if (shardingDomain.type == CommGroupType::None)
        replicationFactor = 1;
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
    // Use the graph associated with RTS indices associated with the groupIndex
    // group
    auto rtsIndices = op_p->getReplicatedTensorShardingIndices();
    ReplicatedTensorShardingIndices::iterator rtsIterator = rtsIndices.begin();
    std::advance(rtsIterator, groupIndex);
    auto p                         = *rtsIterator;
    std::set<InIndex> &inIndices   = p.first;
    std::set<OutIndex> &outIndices = p.second;
    snap::Graph *cbrGraph;
    if (inIndices.size() > 0) {
      cbrGraph = &inGraph(*inIndices.begin());
    } else if (outIndices.size() > 0) {
      cbrGraph = &outGraph(*outIndices.begin());
    } else {
      throw error("[CollectivesBaseOpx::createCollectiveBalancedReorder] "
                  "ReplicatedTensorSharding indices do not correctly map "
                  "to virtual graphs");
    }

    auto cbr = std::make_shared<gcl::CollectiveBalancedReorder>(
        cbrGraph->getPoplarGraph(),
        tensor.getPoplarTensor(),
        replicationFactor,
        getDebugNameAndId(),
        false,
        numGrainElements(cbrGraph->getPoplarGraph().getTarget(),
                         tensor.getPoplarTensor().elementType()));
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

MultiCollectiveBaseOpx::MultiCollectiveBaseOpx(Op *op, Devicex *devicex)
    : CollectivesBaseOpx(op, devicex) {}

std::set<OpxGrowPartId>
MultiCollectiveBaseOpx::getInGrowPartIds(Tensor *inTensor) const {
  std::set<OpxGrowPartId> partIds;
  Op &myOp = getOp<Op>();
  for (auto index : myOp.input->indices(inTensor)) {
    partIds.insert(index % myOp.output->n());
  }
  return partIds;
}

OpxGrowPartId
MultiCollectiveBaseOpx::getOutGrowPartId(Tensor *outTensor) const {
  return getOp<Op>().output->indices(outTensor).at(0);
}

gcl::CommGroup toGCLCommGroup(const popart::CommGroup &group) {
  gcl::CommGroupType type;
  auto size = group.replicaGroupSize;
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
  case popart::CommGroupType::None:
    type = gcl::CommGroupType::CONSECUTIVE;
    size = 1;
    break;
  default:
    throw error("Cannot convert unknown CommGroup type");
  }
  return {type, size};
}

} // namespace popx
} // namespace popart
