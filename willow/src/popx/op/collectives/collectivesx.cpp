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

#include <popops/Collectives.hpp>
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

static int64_t nextMultiple(int64_t val, int64_t multiple) {
  return ((val + multiple - 1) / multiple) * multiple;
}

CollectiveBalancedReorder::CollectiveBalancedReorder(
    poplar::Graph &graph_,
    poplar::Tensor tensor_,
    unsigned replicationFactor_,
    const poplar::DebugNameAndId &dnai_)
    : graph(graph_), replicationFactor(replicationFactor_),
      referenceTensor(tensor_), dnai(dnai_) {
  simplifier    = graph.getSimplifyingRearranger(referenceTensor);
  auto t        = simplifier.rearrange(referenceTensor);
  auto mapping  = graph.getTileMapping(t);
  auto numTiles = mapping.size();

  // Go through each tile of the (potentially simplified) reference
  // tensor and split the contiguous regions between the replicas.
  // Build up a map from indices in the reference tensor to
  // the corresponding replica and index within the replica of
  // the gathered tensor in the 'refToGatheredMap' data structure.
  struct GatherSlice {
    unsigned rep;
    std::size_t index;
    std::size_t size;
    bool operator<(const GatherSlice &other) const {
      return std::tie(rep, index, size) <
             std::tie(other.rep, other.index, other.size);
    }
  };
  numReplicaElementsPerTile.resize(numTiles);
  std::vector<std::pair<std::size_t, GatherSlice>> refToGathered;
  unsigned index = 0;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    auto contRegions   = graph.getSortedContiguousRegions(t, mapping[tile]);
    auto elemsThisTile = poputil::intervalSequenceNumElements(contRegions);
    auto paddedElemsThisTile = nextMultiple(elemsThisTile, replicationFactor);
    numReplicaElementsPerTile[tile] = paddedElemsThisTile / replicationFactor;
    auto perReplicaRegions =
        poputil::splitRegions(contRegions, 1, replicationFactor);

    for (unsigned rep = 0; rep < replicationFactor; ++rep) {
      if (perReplicaRegions.size() <= rep)
        continue;
      auto rIndex = index;
      for (auto &rs : perReplicaRegions[rep]) {
        for (auto &r : rs) {
          refToGathered.emplace_back(r.begin(),
                                     GatherSlice{rep, rIndex, r.size()});
          rIndex += r.size();
        }
      }
    }
    index += numReplicaElementsPerTile[tile];
  }
  hostRearrangement.replicationFactor       = replicationFactor_;
  hostRearrangement.totalElementsPerReplica = index;

  std::sort(refToGathered.begin(), refToGathered.end());

  // Now create the list of intervals in the gathered tensor
  // that make up a view of the (simplified) reference tensor.
  for (const auto &entry : refToGathered) {
    const auto begin =
        entry.second.rep * hostRearrangement.totalElementsPerReplica +
        entry.second.index;
    gatheredToSimplifiedRefSlices.emplace_back(begin,
                                               begin + entry.second.size);
  }

  // We need a second set of intervals that goes directly from
  // the gathered tensor ordering -> reference tensor ordering
  // for CollectiveBalancedReorder::rearrange

  // First we get an ordering of slices in the simplified ordering using
  // the rearranger. For convenience later we ensure this is split by the
  // intervals in the gathered -> simplified map.
  std::vector<poplar::Interval> simplifiedToRefSlices;
  simplifiedToRefSlices.reserve(refToGathered.size());
  std::transform(refToGathered.begin(),
                 refToGathered.end(),
                 std::back_inserter(simplifiedToRefSlices),
                 [](const auto &x) {
                   return poplar::Interval(x.first, x.first + x.second.size);
                 });
  simplifiedToRefSlices = simplifier.undoRearrangement(simplifiedToRefSlices);
  const auto numSimplifiedRefSlices = simplifiedToRefSlices.size();

  // Now we need to apply the same reordering as above to the intervals which
  // go from gathered -> simplified so that we have intervals that go from
  // gathered -> reference tensor ordering.
  std::vector<std::size_t> simplifiedOrderingIndices(numSimplifiedRefSlices);
  std::iota(
      simplifiedOrderingIndices.begin(), simplifiedOrderingIndices.end(), 0);
  std::sort(simplifiedOrderingIndices.begin(),
            simplifiedOrderingIndices.end(),
            [&](const auto a, const auto b) {
              return simplifiedToRefSlices[a] < simplifiedToRefSlices[b];
            });

  hostRearrangement.gatheredToRefSlices.resize(numSimplifiedRefSlices);
  auto it = gatheredToSimplifiedRefSlices.begin();
  std::size_t gatheredToSimplifiedRefOffset = 0;
  for (const auto i : simplifiedOrderingIndices) {
    const auto &simplifiedToRefSlice = simplifiedToRefSlices[i];

    const auto begin = it->begin() + gatheredToSimplifiedRefOffset;
    const auto end   = begin + simplifiedToRefSlice.size();

    hostRearrangement.gatheredToRefSlices[i] = poplar::Interval(begin, end);

    gatheredToSimplifiedRefOffset += simplifiedToRefSlice.size();
    if (gatheredToSimplifiedRefOffset == it->size()) {
      ++it;
      gatheredToSimplifiedRefOffset = 0;
    }
  }
}

poplar::Tensor
CollectiveBalancedReorder::createReplicaSlice(const poplar::Type &type) {
  // A replica slice is a single variable with the tile
  // mapping set so you get a contiguous region on each
  // tile of the correct size to map the reference tensor to.
  auto t = graph.addVariable(
      type, {hostRearrangement.totalElementsPerReplica}, {dnai, "_cbr_slice0"});
  auto index = 0;
  for (unsigned tile = 0; tile < numReplicaElementsPerTile.size(); ++tile) {
    auto size = numReplicaElementsPerTile[tile];
    graph.setTileMapping(t.slice(index, index + size), tile);
    index += size;
  }
  return t;
}

poplar::Tensor CollectiveBalancedReorder::createCollectivesTensor(
    const poplar::Type &type,
    const std::string &debugPrefix) {
  // The full collectives (gathered) tensor is just the
  // concatenation of 'replicaFactor' slices.
  std::vector<poplar::Tensor> slices = {createReplicaSlice(type).expand({0})};
  for (unsigned i = 1; i < replicationFactor; ++i) {
    auto name = debugPrefix + "_cbr_slice" + std::to_string(i);
    slices.push_back(graph.clone(slices[0], {dnai, name}));
  }
  return concat(slices);
}

poplar::Tensor CollectiveBalancedReorder::undoRearrangeForCollective(
    poplar::Tensor tensor) const {
  // To go from a gathered tensor to a view that looks like the
  // reference tensor we use the list of regions in
  // 'gatheredToSimplifiedRefSlices' to get a view with the ordering
  // of the simplified reference tensor, then we use the simplifier
  // to get a view with the ordering of the original reference tensor.
  auto t = concat(tensor.flatten().slices(gatheredToSimplifiedRefSlices));
  t      = simplifier.undoRearrangement(t);
  return t.reshape(referenceTensor.shape());
}

size_t
CollectiveBalancedHostRearrangement::getNumRearrangedTensorElems() const {
  return totalElementsPerReplica * replicationFactor;
}

void CollectiveBalancedHostRearrangement::rearrange(const char *in,
                                                    char *out,
                                                    int64_t elemByteSize,
                                                    bool refToGathered) const {
  auto index = 0;
  for (const auto &i : gatheredToRefSlices) {
    auto size    = i.size();
    auto gOffset = i.begin() * elemByteSize;
    auto rOffset = index * elemByteSize;
    if (refToGathered) {
      std::memcpy(out + gOffset, in + rOffset, size * elemByteSize);
    } else {
      std::memcpy(out + rOffset, in + gOffset, size * elemByteSize);
    }
    index += size;
  }
}

void CollectiveBalancedHostRearrangement::rearrangeForCollective(
    const char *in,
    char *out,
    int64_t elemByteSize) const {
  rearrange(in, out, elemByteSize, true);
}

void CollectiveBalancedHostRearrangement::undoRearrangeForCollective(
    const char *in,
    char *out,
    int64_t elemByteSize) const {
  rearrange(in, out, elemByteSize, false);
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

CollectiveBalancedReorder *
CollectivesBaseOpx::getCollectiveBalancedReorder() const {
  auto group = getCollectiveLinkedGroup();
  logging::opx::trace("[CollectivesBaseOpx] Getting CBR for {}",
                      *group.first.begin());
  auto cbr =
      dv_p->lowering().getCollectiveBalancedReorder(*group.first.begin());
  return cbr.get();
}

CollectiveBalancedReorder *CollectivesBaseOpx::createCollectiveBalancedReorder(
    poplar::Tensor tensor) const {
  auto replicationFactor = dv_p->lowering().getReplicationFactor();
  auto group             = getCollectiveLinkedGroup();
  auto cbr =
      dv_p->lowering().getCollectiveBalancedReorder(*group.first.begin());
  if (!cbr.get()) {
    cbr = std::make_shared<CollectiveBalancedReorder>(
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
