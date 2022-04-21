// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <functional>
#include <memory>
#include <onnx/onnx_pb.h>
#include <popart/graph.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/opserialiser.hpp>
#include <popart/scope.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

bool SubgraphOp::existsInBodyInputs(std::vector<std::string> &bodyInputIds,
                                    TensorId &tensorId) {
  auto found =
      std::find(std::begin(bodyInputIds), std::end(bodyInputIds), tensorId);
  return found != std::end(bodyInputIds);
}

bool SubgraphOp::existsInOpInputs(
    std::vector<std::pair<TensorId, TensorInfo>> &opInputs,
    TensorId &tensorId) {
  auto found =
      std::find_if(std::begin(opInputs),
                   std::end(opInputs),
                   [&tensorId](const std::pair<TensorId, TensorInfo> &kv) {
                     return kv.first == tensorId;
                   });
  return found != std::end(opInputs);
}

std::vector<TensorId>
SubgraphOp::getBodyInputIds(const ONNX_NAMESPACE::GraphProto &bodyProto) {
  std::vector<TensorId> bodyInputs;
  for (int i = 0; i < bodyProto.input_size(); ++i) {
    bodyInputs.push_back(bodyProto.input(i).name());
  }
  return bodyInputs;
}

std::vector<TensorId>
SubgraphOp::getBodyOutputIds(const ONNX_NAMESPACE::GraphProto &bodyProto) {
  std::vector<TensorId> bodyOutputs;
  for (int i = 0; i < bodyProto.output_size(); ++i) {
    bodyOutputs.push_back(bodyProto.output(i).name());
  }
  return bodyOutputs;
}

SubgraphOp::SubgraphOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings_)
    : Op(_opid, settings_), calledGraphGradOpHelper{this} {}

void SubgraphOp::addAlias(InIndex in,
                          OutIndex out,
                          view::Chains fwdChains,
                          view::Chains bwdChains) {
  aliasMap.insert(std::make_pair(std::make_pair(in, out),
                                 std::make_pair(fwdChains, bwdChains)));
}

void SubgraphOp::adjustAliasInIndices(InIndex fromIn, InIndex toIn) {
  // TODO(T43804): This function iterates over all items in `aliasMap`, which
  // makes it quite slow. It can be done in constant time with some new data
  // structures.
  std::map<std::pair<InIndex, OutIndex>, std::pair<view::Chains, view::Chains>>
      updateMap;

  // Remove old aliases from map.
  for (auto it = aliasMap.cbegin(); it != aliasMap.cend();) {
    const auto &in  = it->first.first;
    const auto &out = it->first.second;
    if (in == fromIn) {
      updateMap[{toIn, out}] = it->second;
      it                     = aliasMap.erase(it);
    } else {
      ++it;
    }
  }

  // Update the map with the new aliases.
  aliasMap.insert(updateMap.begin(), updateMap.end());
}

void SubgraphOp::adjustAliasOutIndices(OutIndex fromOut, OutIndex toOut) {
  // TODO(T43804): This function iterates over all items in `aliasMap`, which
  // makes it quite slow. It can be done in constant time with some new data
  // structures.
  std::map<std::pair<InIndex, OutIndex>, std::pair<view::Chains, view::Chains>>
      updateMap;

  // Remove old aliases from map.
  for (auto it = aliasMap.cbegin(); it != aliasMap.cend();) {
    const auto &in  = it->first.first;
    const auto &out = it->first.second;
    if (out == fromOut) {
      updateMap[{in, toOut}] = it->second;
      it                     = aliasMap.erase(it);
    } else {
      ++it;
    }
  }

  // Update the map with the new aliases.
  aliasMap.insert(updateMap.begin(), updateMap.end());
}

void SubgraphOp::adjustModifiedIndices(InIndex fromIn, InIndex toIn) {
  if (modifiesMap.count(fromIn) != 0) {
    const auto &regions = modifiesMap[fromIn];
    modifiesMap[toIn]   = regions;
    modifiesMap.erase(fromIn);
  }
}

void SubgraphOp::addModified(InIndex in, view::Regions regions) {
  modifiesMap.insert(std::make_pair(in, regions));
}

void SubgraphOp::removeModified(InIndex in) { modifiesMap.erase(in); }

void SubgraphOp::removeAlias(InIndex in, OutIndex out) {
  aliasMap.erase({in, out});
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

std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
SubgraphOp::fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                                       const ReplEqInputMap &opInputMap,
                                       ReplicaEqualAnalysisProxy &proxy) const {

  // Map Op inputs mapping to subgraph input mapping.
  ReplEqInputMap subgraphInputMap;
  for (InIndex i = 0; i < getCalledGraph().getInputIds().size(); ++i) {
    subgraphInputMap[i] = opInputMap.at(subgraphInToOpInIndex(i));
  }

  // Forward propagate on subgraph.
  auto subgraphRes = proxy.fwdPropagateIsReplicaEqualThroughGraph(
      &getCalledGraph(), subgraphInputMap);
  auto &subgraphOutputMap         = std::get<0>(subgraphRes);
  auto &subgraphModifiedInputsMap = std::get<1>(subgraphRes);

  // Map Op output mapping back to subgraph output mapping.
  ReplEqInputMap opOutputMap;
  for (auto &entry : output->tensorMap()) {
    OutIndex o     = entry.first;
    opOutputMap[o] = subgraphOutputMap.at(opOutToSubgraphOutIndex(o));
  }

  // Get replica-equalness of modified inputs from subgraph.
  ReplEqModifiedInputMap modifiedInputs;
  for (const auto &input : input->tensorMap()) {
    // Ignore this index if it is not modified.
    if (modifiesIndex(input.first)) {
      modifiedInputs[input.first] =
          subgraphModifiedInputsMap.at(opInToSubgraphInIndex(input.first));
    }
  }

  return {opOutputMap, modifiedInputs};
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
      throw error("Invalid Region of rank {} in SubgraphOp::aliases at InIndex "
                  "{} where the input Tensor is of rank {}.",
                  r.rank(),
                  in,
                  inRank(in));
    }
  }

  return aliasRegions;
}

void SubgraphOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);

  {
    std::stringstream ss;
    ss << "_aliases[";
    for (auto &alias : aliasMap) {

      auto t0 = inTensor(alias.first.first);
      auto t1 = outTensor(alias.first.second);

      auto fullRegion0 = view::Region::getFull(t0->info.shape());
      auto regions0    = alias.second.first.apply(fullRegion0);
      auto fullRegion1 = view::Region::getFull(t1->info.shape());
      auto regions1    = alias.second.second.apply(fullRegion1);

      ss << "(" << alias.first.first << ":" << alias.first.second << "):"
         << "(" << regions0 << ":" << regions1 << ")";
    }
    ss << "]";
    os.appendAttribute("aliases", ss.str());
  }

  {
    std::stringstream ss;
    ss << "_modifies[";
    for (auto &modifies : modifiesMap) {
      if (std::any_of(modifies.second.begin(),
                      modifies.second.end(),
                      [](const view::Region &r) { return !r.isEmpty(); })) {
        ss << "(" << modifies.first;
        ss << ":";
        ss << modifies.second << ")";
      }
    }
    ss << "]";
    os.appendAttribute("modifies", ss.str());
  }
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
      throw error("Invalid Region of rank {} in SubgraphOp::modifies at InIndex"
                  " {} where the input Tensor is of rank {}.",
                  r.rank(),
                  in,
                  inRank(in));
    }
  }

  return modifiedRegions;
}

VGraphIdAndTileSet
SubgraphOp::getIntrospectionInVirtualGraphId(InIndex index,
                                             std::set<OpId> &visited) const {
  visited.insert(id);

  std::set<VGraphIdAndTileSet, VGraphIdAndTileSetCmp> vgidSet;

  InIndex subgraphInIndex = opInToSubgraphInIndex(index);

  if (subgraphInIndex > -1 &&
      subgraphInIndex < getCalledGraph().getInputIds().size()) {
    auto tensor_id = getCalledGraph().getInputId(subgraphInIndex);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // The tensor knows it's own VGID
    auto vgid = tensor->getVirtualGraphIdAndTileSetUnsafe(visited);
    vgidSet.insert(vgid);

    // The tensor has no consumer inside the graph,
    // look after the graph
    if (tensor->consumers.getOps().empty() && tensor->isGraphOutput()) {
      OutIndex sgOutIndex = tensor->getGraphOutputIndex();
      OutIndex opOutIndex = subgraphOutToOpOutIndex(sgOutIndex);
      if (output->hasIndex(opOutIndex)) {
        auto vgid = output->tensor(opOutIndex)
                        ->getVirtualGraphIdAndTileSetUnsafe(visited);
        vgidSet.insert(vgid);
      }
    }
  } else {
    // No VGID determined by introspection or tensor
    if (Op::hasVirtualGraphId()) {
      vgidSet.insert({Op::getVirtualGraphId(), getSettings().tileSet});
    }
  }

  if (vgidSet.empty()) {
    // No virtual graph ID and tile set determined
    vgidSet.insert({unusedVGraphId, TileSet::Undefined});
  }

  return *vgidSet.begin();
}

VGraphIdAndTileSet
SubgraphOp::getIntrospectionOutVirtualGraphId(OutIndex index,
                                              std::set<OpId> &visited) const {
  visited.insert(id);

  std::set<VGraphIdAndTileSet, VGraphIdAndTileSetCmp> vgidSet;

  InIndex subgraphOutIndex = opOutToSubgraphOutIndex(index);

  if (subgraphOutIndex > -1 &&
      subgraphOutIndex < getCalledGraph().getOutputIds().size()) {
    auto tensor_id = getCalledGraph().getOutputId(subgraphOutIndex);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // The tensor knows it's own VGID
    auto vgid = tensor->getVirtualGraphIdAndTileSetUnsafe(visited);
    vgidSet.insert(vgid);

    // The tensor has no producer inside the graph,
    // look before the graph
    if (tensor->isGraphInput()) {
      InIndex sgInIndex = tensor->getGraphInputIndex();
      InIndex opInIndex = subgraphInToOpInIndex(sgInIndex);
      if (hasInput(opInIndex)) {
        auto vgid = input->tensor(opInIndex)->getVirtualGraphIdAndTileSetUnsafe(
            visited);
        vgidSet.insert(vgid);
      }
    }

  } else {
    // No VGID determined by introspection or tensor
    if (Op::hasVirtualGraphId()) {
      vgidSet.insert({Op::getVirtualGraphId(), getSettings().tileSet});
    }
  }

  if (vgidSet.empty()) {
    // No virtual graph ID and tile set determined
    vgidSet.insert({unusedVGraphId, TileSet::Undefined});
  }

  return *vgidSet.begin();
}

bool SubgraphOp::hasSideEffect() const {
  auto &ops = getCalledGraph().getOps();
  return std::any_of(ops.begin(), ops.end(), [](const auto &op) {
    return op.second->hasSideEffect();
  });
}

std::vector<const Graph *> SubgraphOp::getCalledGraphs() const {
  std::vector<const Graph *> calledGraphs;
  calledGraphs.push_back(&getCalledGraph());
  return calledGraphs;
}

InIndex SubgraphOp::opInToSubgraphInIndex(SubgraphIndex subgraphIndex,
                                          InIndex inIndex) const {
  if (subgraphIndex != 0) {
    throw error("Invalid subgraphIndex for Op {} (expected 0, got {})",
                debugName(),
                subgraphIndex);
  }

  if (!input->hasIndex(inIndex)) {
    throw error(
        "Invalid inIndex for Op {} (op does not have an input with index {})",
        debugName(),
        inIndex);
  }

  return opInToSubgraphInIndex(inIndex);
}

InIndex SubgraphOp::subgraphInToOpInIndex(SubgraphIndex subgraphIndex,
                                          InIndex inIndex) const {
  if (subgraphIndex != 0) {
    throw error("Invalid subgraphIndex for Op {} (expected 0, got {})",
                debugName(),
                subgraphIndex);
  }

  if (inIndex < 0 || inIndex >= getCalledGraph().getInputIds().size()) {
    throw error("Invalid inIndex for subgraph '{}' (subgraph does not have an "
                "input with index {})",
                getCalledGraph().id.str(),
                inIndex);
  }

  return subgraphInToOpInIndex(inIndex);
}

OutIndex SubgraphOp::opOutToSubgraphOutIndex(SubgraphIndex subgraphIndex,
                                             OutIndex outIndex) const {
  if (subgraphIndex != 0) {
    throw error("Invalid subgraphIndex for Op {} (expected 0, got {})",
                debugName(),
                subgraphIndex);
  }

  if (!output->hasIndex(outIndex)) {
    throw error(
        "Invalid outIndex for Op {} (op does not have an output with index {})",
        debugName(),
        outIndex);
  }

  return opOutToSubgraphOutIndex(outIndex);
}

OutIndex SubgraphOp::subgraphOutToOpOutIndex(SubgraphIndex subgraphIndex,
                                             OutIndex outIndex) const {
  if (subgraphIndex != 0) {
    throw error("Invalid subgraphIndex for Op {} (expected 0, got {})",
                debugName(),
                subgraphIndex);
  }

  if (outIndex < 0 || outIndex >= getCalledGraph().getOutputIds().size()) {
    throw error("Invalid inIndex for subgraph '{}' (subgraph does not have an "
                "output with index {})",
                getCalledGraph().id.str(),
                outIndex);
  }

  return subgraphOutToOpOutIndex(outIndex);
}

float SubgraphOp::calcAutoVirtualGraphCost(std::set<int> &inputs_seen) {
  return 0.0f;
}

void SubgraphOp::setCalledSubgraphGradInfo(
    const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {
  calledGraphGradOpHelper.setCalledSubgraphGradInfo(calledGraphsGradInfo);
}

} // namespace popart
