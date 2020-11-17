// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/opserialiser.hpp>
#include <popart/scope.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

SubgraphOp::SubgraphOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void SubgraphOp::addAlias(InIndex in,
                          OutIndex out,
                          view::Chains fwdChains,
                          view::Chains bwdChains) {
  aliasMap.insert(std::make_pair(std::make_pair(in, out),
                                 std::make_pair(fwdChains, bwdChains)));
}

void SubgraphOp::addModified(InIndex in, view::Regions regions) {
  modifiesMap.insert(std::make_pair(in, regions));
}

view::RegMap SubgraphOp::fwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  auto emptyRegion = view::Region::getEmpty(outRank(outIndex));
  auto aliasChains = aliasMap.at({inIndex, outIndex}).first;

  // Only capture by value
  return [aliasChains, emptyRegion](const view::Region &r) {
    if (r.isEmpty() || aliasChains.isEmpty()) {
      return view::Regions(1, emptyRegion);
    } else {
      return aliasChains.apply(r);
    }
  };
}

view::RegMap SubgraphOp::bwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  auto emptyRegion = view::Region::getEmpty(inRank(inIndex));
  auto aliasChains = aliasMap.at({inIndex, outIndex}).second;

  // Only capture by value
  return [aliasChains, emptyRegion](const view::Region &r) {
    if (r.isEmpty() || aliasChains.isEmpty()) {
      return view::Regions(1, emptyRegion);
    } else {
      return aliasChains.apply(r);
    }
  };
}

view::Regions SubgraphOp::aliases(InIndex in, OutIndex out) const {
  // If not in aliasMap, return empty region
  if (aliasMap.count({in, out}) == 0) {
    return {view::Region::getEmpty(inRank(in))};
  }

  // Regions of in which are aliased
  auto aliasRegions =
      aliasMap.at({in, out}).second.apply(view::Region::getFull(outShape(out)));
  for (const auto &r : aliasRegions) {
    if (r.rank() != inRank(in)) {
      throw error("Invalid Region of rank {} in CallOp::aliases at InIndex {} "
                  "where the input Tensor is of rank {}.",
                  r.rank(),
                  in,
                  inRank(in));
    }
  }

  return aliasRegions;
}

view::Regions SubgraphOp::modifies(InIndex in) const {
  // If not in modifiesMap, return empty region
  if (modifiesMap.count(in) == 0) {
    return {view::Region::getEmpty(inRank(in))};
  }

  // Regions of in which are aliased
  auto modifiedRegions = modifiesMap.at(in);
  for (const auto &r : modifiedRegions) {
    if (r.rank() != inRank(in)) {
      throw error("Invalid Region of rank {} in CallOp::modifies at InIndex {} "
                  "where the input Tensor is of rank {}.",
                  r.rank(),
                  in,
                  inRank(in));
    }
  }

  return modifiedRegions;
}

VGraphIdAndTileSet
SubgraphOp::getIntrospectionInVirtualGraphId(InIndex index) const {

  InIndex subgraphInIndex = opInToSubgraphInIndex(index);

  if (subgraphInIndex > -1) {
    auto num_ids = getCalledGraph().getInputIds().size();
    if (subgraphInIndex >= num_ids)
      throw error("[getIntrospectionInVirtualGraphId] "
                  "SubgraphOp ({}) has {} subgraph inputs, "
                  "but requested index is {}",
                  debugName(),
                  num_ids,
                  subgraphInIndex);

    auto tensor_id = getCalledGraph().getInputId(subgraphInIndex);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // Callee introspection
    for (auto consumer : tensor->consumers.getOps()) {
      if (dynamic_cast<SubgraphOp *>(consumer)) {
        auto subindex = consumer->input->indicesMap().at(tensor)[0];
        if (consumer->hasVirtualGraphId()) {
          // Also works if the callee is another subgraph
          auto intropId = consumer->getIntrospectionInVirtualGraphId(subindex);
          if (intropId.first > -1)
            return intropId;
        }
        if (IpuCopyOp *copyConsumer = dynamic_cast<IpuCopyOp *>(consumer)) {
          return {copyConsumer->getSourceIpu(tensor_id),
                  copyConsumer->settings.tileSet};
        }
      }
    }

    // Fallback 1: The tensor knows it's own VGID
    // We ask this only after callee introspection, because otherwise the
    // CallOp's VGID will be reported, which can be wrong if it's nested
    // consuming operator is on another virtual graph.
    if (tensor->hasVirtualGraphId()) {
      // Tensor has VirtualGraphID given by it's producer or consumer
      auto vgId = tensor->getVirtualGraphIdAndTileSet();
      if (vgId.first > -1) {
        return vgId;
      }
    }
  }

  // Fallback 2: No VGID determined by introspection or tensor
  return Op::hasVirtualGraphId()
             ? VGraphIdAndTileSet(Op::getVirtualGraphId(),
                                  getSettings().tileSet)
             : VGraphIdAndTileSet(unusedVGraphId, TileSet::Compute);
}

VGraphIdAndTileSet
SubgraphOp::getIntrospectionOutVirtualGraphId(OutIndex index) const {

  InIndex subgraphOutIndex = opOutToSubgraphOutIndex(index);

  if (subgraphOutIndex > -1) {
    auto num_ids = getCalledGraph().getOutputIds().size();
    if (subgraphOutIndex >= num_ids)
      throw error("[getIntrospectionOutVirtualGraphId] "
                  "SubgraphOp ({}) has {} subgraph inputs, "
                  "but requested index is {}",
                  debugName(),
                  num_ids,
                  subgraphOutIndex);

    auto tensor_id = getCalledGraph().getOutputId(subgraphOutIndex);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // Callee introspection
    auto producer = tensor->getProducer();
    if (dynamic_cast<SubgraphOp *>(producer)) {
      auto subindex = producer->output->indicesMap().at(tensor)[0];
      if (producer->hasVirtualGraphId()) {
        // Also works if the callee is another subgraph
        auto vgId = producer->getIntrospectionOutVirtualGraphId(subindex);
        if (vgId.first > -1) {
          return vgId;
        }
      }
    }

    // Fallback 1: The tensor knows it's own VGID
    // We ask this only after callee introspection, because otherwise the
    // CallOp's VGID will be reported, which can be wrong if it's nested
    // consuming operator is on another virtual graph.
    if (tensor->hasVirtualGraphId()) {
      // Tensor has VirtualGraphID given by it's producer or consumer
      auto vgId = tensor->getVirtualGraphIdAndTileSet();
      if (vgId.first > -1) {
        return vgId;
      }
    }
  }

  // Fallback 2: No VGID determined by introspection or tensor
  return Op::hasVirtualGraphId()
             ? VGraphIdAndTileSet(Op::getVirtualGraphId(),
                                  getSettings().tileSet)
             : VGraphIdAndTileSet(unusedVGraphId, TileSet::Compute);
}

} // namespace popart
