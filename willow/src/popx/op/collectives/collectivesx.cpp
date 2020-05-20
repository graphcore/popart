// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Collectives.hpp>

namespace popart {
namespace popx {

CollectiveBalancedReorder::CollectiveBalancedReorder(
    poplar::Graph &graph_,
    poplar::Tensor tensor_,
    unsigned replicationFactor_)
    : graph(graph_), replicationFactor(replicationFactor_) {
  referenceTensor = graph.clone(tensor_);
  elemByteSize = graph.getTarget().getTypeSize(referenceTensor.elementType());
  auto flat    = tensor_.flatten();

  int64_t minRegionSize = 1;

  auto nextMultiple = [](int64_t val, int64_t multiple) {
    return ((val + multiple - 1) / multiple) * multiple;
  };

  simplifyProxy = graph.addVariable(flat.elementType(), {flat.numElements()});
  simplifyReverseProxy =
      graph.addVariable(flat.elementType(), {flat.numElements()});
  graph.setTileMapping(simplifyProxy, 0);
  graph.setTileMapping(simplifyReverseProxy, 0);

  graph.reorderToSimplify(&flat, {&simplifyReverseProxy});
  poplar::Tensor reverse = simplifyReverseProxy;
  graph.reorderToSimplify(&reverse, {&simplifyProxy});

  auto mapping = graph.getTileMapping(flat);

  std::vector<std::vector<poplar::Tensor>> reorderedSlices(replicationFactor);
  std::vector<size_t> reloc(replicationFactor);

  std::vector<std::vector<std::vector<poplar::Interval>>> contRegionsPerTile;

  for (const auto &m : mapping) {
    contRegionsPerTile.emplace_back(graph.getSortedContiguousRegions(flat, m));
  }

  size_t numElemsPerReplica = 0;
  for (unsigned tile = 0; tile < contRegionsPerTile.size(); ++tile) {
    auto contRegions = contRegionsPerTile.at(tile);
    if (contRegions.size() > 0) {
      for (auto contRegion : contRegions) {
        size_t totalSize = 0;
        for (auto region : contRegion) {
          totalSize += region.size();
        }
        numElemsPerReplica += nextMultiple(
            (totalSize - 1) / replicationFactor + 1, minRegionSize);
      }
    }
  }

  for (unsigned i = 0; i < replicationFactor; ++i) {
    reloc[i] = i * numElemsPerReplica;
  }

  for (unsigned tile = 0; tile < contRegionsPerTile.size(); ++tile) {
    auto contRegions = contRegionsPerTile.at(tile);

    if (contRegions.size() > 0) {
      for (auto contRegion : contRegions) {
        size_t totalSize = 0;
        for (auto region : contRegion) {
          totalSize += region.size();
        }
        size_t totalReplicaSize = nextMultiple(
            (totalSize - 1) / replicationFactor + 1, minRegionSize);

        size_t regionIndex  = 0;
        size_t regionOffset = 0;
        size_t replicaIndex = 0;
        size_t replicaSize  = 0;
        while (replicaIndex < replicationFactor) {
          if (regionIndex < contRegion.size()) {
            auto &region = contRegion[regionIndex];

            auto size = std::min(region.size() - regionOffset,
                                 totalReplicaSize - replicaSize);

            size_t start = region.begin() + regionOffset;

            // Slice of the tensor
            reordering.emplace_back(start, reloc[replicaIndex], size, tile);

            reloc[replicaIndex] += size;
            regionOffset += size;
            replicaSize += size;

            if (regionOffset == region.size()) {
              ++regionIndex;
              regionOffset = 0;
            }
          } else {
            auto size = totalReplicaSize - replicaSize;
            // Padding
            reordering.emplace_back(-1, reloc[replicaIndex], size, tile);
            reloc[replicaIndex] += size;
            replicaSize += size;
          }
          if (replicaSize == totalReplicaSize) {
            ++replicaIndex;
            replicaSize = 0;
          }
        }
      }
    }
  }
  numRearrangedTensorElems =
      reordering.back().rearranged_offset + reordering.back().size;
}

poplar::Tensor
CollectiveBalancedReorder::rearrangeForCollective(poplar::Tensor tensor) const {

  auto reorder = reordering;

  auto flat = tensor.flatten();

  // Simplify
  poplar::Tensor simplify = simplifyProxy;
  graph.reorderToSimplify(&simplify, {&flat});

  // Sort by start offset in the rearranged tensor
  std::sort(reorder.begin(),
            reorder.end(),
            [](const ReorderMetadata &a, const ReorderMetadata &b) {
              return a.rearranged_offset < b.rearranged_offset;
            });

  std::vector<poplar::Tensor> allReorderedSlices;

  size_t padSize = 0;
  for (auto &r : reorder) {
    if (r.offset == -1) {
      padSize += r.size;
    }
  }

  poplar::Tensor pad = graph.addVariable(flat.elementType(), {padSize});
  int64_t padOffset  = 0;

  // Per-replica reordering
  for (auto &r : reorder) {
    if (r.offset > -1) {
      allReorderedSlices.push_back(flat.slice(r.offset, r.offset + r.size));
    } else {
      allReorderedSlices.push_back(pad.slice(padOffset, padOffset + r.size));
      graph.setTileMapping(allReorderedSlices.back(),
                           static_cast<unsigned>(r.tile));
      padOffset += r.size;
    }
  }

  poplar::Tensor concatResult = poplar::concat(allReorderedSlices);
  return concatResult;
}

poplar::Tensor CollectiveBalancedReorder::undoRearrangeForCollective(
    poplar::Tensor tensor) const {

  auto reorder = reordering;

  auto flat = tensor.flatten();

  // Sort by start offset in the original flat tensor
  std::sort(reorder.begin(),
            reorder.end(),
            [](const ReorderMetadata &a, const ReorderMetadata &b) {
              return a.offset < b.offset;
            });

  std::vector<poplar::Tensor> allReorderedSlices;

  // Reverse per-replica reordering
  for (auto &r : reorder) {
    // Skip padding
    if (r.offset > -1) {
      allReorderedSlices.push_back(
          flat.slice(r.rearranged_offset, r.rearranged_offset + r.size));
    }
  }

  poplar::Tensor concatResult = poplar::concat(allReorderedSlices);
  poplar::Tensor reverse      = simplifyReverseProxy;

  // Reverse simplification
  graph.reorderToSimplify(&reverse, {&concatResult});

  return concatResult;
}

void CollectiveBalancedReorder::rearrange(const char *in,
                                          char *out,
                                          bool forCollective) const {
  auto reorder = reordering;

  // Sort by start offset in the original tensor
  std::sort(reorder.begin(),
            reorder.end(),
            [](const ReorderMetadata &a, const ReorderMetadata &b) {
              return a.offset < b.offset;
            });

  // Offsets in the original tensor, sorted by the simplified tensor order
  auto intervals = graph.getSortedContiguousRegions(
      simplifyProxy, graph.getTileMapping(simplifyProxy)[0])[0];

  int64_t intervalIndex  = 0;
  int64_t intervalOffset = 0;

  for (auto &r : reorder) {
    if (r.offset > -1) {
      // Translate offset in the simplifed tensor (ostart) to offset in the
      // original input tensor (osstart)
      int64_t copiedOffset = 0;
      while (copiedOffset < r.size) {
        auto currentInterval = intervals[intervalIndex];
        int64_t osstart      = currentInterval.begin();
        int64_t intervalSize = currentInterval.size();
        int64_t size =
            std::min(intervalSize - intervalOffset, r.size - copiedOffset);

        int64_t inOff;
        int64_t outOff;
        if (forCollective) {
          inOff  = (osstart + intervalOffset) * elemByteSize;
          outOff = (r.rearranged_offset + copiedOffset) * elemByteSize;
        } else {
          outOff = (osstart + intervalOffset) * elemByteSize;
          inOff  = (r.rearranged_offset + copiedOffset) * elemByteSize;
        }

        auto byteSize = size * elemByteSize;

        std::memcpy(out + outOff, in + inOff, byteSize);

        copiedOffset += size;
        intervalOffset += size;
        if (intervalOffset == currentInterval.size()) {
          // Next proxy interval
          ++intervalIndex;
          intervalOffset = 0;
        }
      }
    }
  }
}

void CollectiveBalancedReorder::rearrangeForCollective(const char *in,
                                                       char *out) const {
  rearrange(in, out, true);
}

void CollectiveBalancedReorder::undoRearrangeForCollective(const char *in,
                                                           char *out) const {
  rearrange(in, out, false);
}

poplar::Tensor
CollectiveBalancedReorder::getReferenceTensorClone(std::string name) const {
  return graph.clone(referenceTensor, name);
}

const poplar::Tensor &CollectiveBalancedReorder::getReferenceTensor() const {
  return referenceTensor;
}

CollectivesBaseOpx::CollectivesBaseOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex){};

std::pair<std::set<TensorId>, std::vector<Op *>>
CollectivesBaseOpx::getCollectiveLinkedGroup() const {
  // Mapping from each CacheArg to it's final consumers
  std::map<TensorId, std::set<Op *>> linkOpMap;
  std::map<Op *, std::set<TensorId>> opLinkMap;

  Ir &ir                                     = op_p->getIr();
  const liveness::LivenessAnalyzer *liveness = dv_p->getLivenessAnalyzer();

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
          throw error("Op {} not expected on the path from the"
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
  auto cbr = dv_p->getCollectiveBalancedReorder(*group.first.begin());
  return cbr.get();
}

CollectiveBalancedReorder *CollectivesBaseOpx::createCollectiveBalancedReorder(
    poplar::Tensor tensor) const {
  auto replicationFactor = dv_p->getReplicationFactor();
  auto group             = getCollectiveLinkedGroup();
  auto cbr = dv_p->getCollectiveBalancedReorder(*group.first.begin());
  if (!cbr.get()) {
    cbr = std::make_shared<CollectiveBalancedReorder>(
        graph(), tensor, replicationFactor);
    for (auto tensor_id : group.first) {
      logging::opx::trace("[CollectivesBaseOpx] CBR created for {}", tensor_id);
      dv_p->setCollectiveBalancedReorder(tensor_id, cbr);
    }
  }
  return cbr.get();
}

} // namespace popx
} // namespace popart
