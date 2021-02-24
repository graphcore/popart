// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

std::vector<TensorId> SubgraphOp::getImplicitTensors(
    const ONNX_NAMESPACE::GraphProto &bodyProto,
    popart::Tensors &tensors,
    std::vector<std::pair<TensorId, TensorInfo>> &allOpInputs) {

  auto bodyInputIds = SubgraphOp::getBodyInputIds(bodyProto);
  std::vector<TensorId> implicitTensors;

  for (int i = 0; i < bodyProto.node_size(); ++i) {
    auto &nodeProto = bodyProto.node(i);
    for (int j = 0; j < nodeProto.input_size(); ++j) {
      auto tid        = nodeProto.input(j);
      auto inLoopBody = SubgraphOp::existsInBodyInputs(bodyInputIds, tid);
      if (!inLoopBody) {
        auto inOpInputs = SubgraphOp::existsInOpInputs(allOpInputs, tid);
        if (!inOpInputs) {
          if (tensors.contains(tid)) {
            implicitTensors.push_back(tid);
            allOpInputs.push_back(std::make_pair(tid, tensors.get(tid)->info));
          }
        }
      }
    }
  }

  return implicitTensors;
}

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
SubgraphOp::getIntrospectionInVirtualGraphId(InIndex index,
                                             std::set<OpId> visited) const {
  visited.insert(id);

  InIndex subgraphInIndex = opInToSubgraphInIndex(index);

  if (subgraphInIndex > -1) {
    auto num_ids = getCalledGraph().getInputIds().size();
    if (subgraphInIndex >= num_ids) {
      throw error("[getIntrospectionInVirtualGraphId] "
                  "SubgraphOp ({}) has {} subgraph inputs, "
                  "but requested index is {}",
                  debugName(),
                  num_ids,
                  subgraphInIndex);
    }

    auto tensor_id = getCalledGraph().getInputId(subgraphInIndex);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // Callee introspection
    for (auto consumer : tensor->consumers.getOps()) {
      if (visited.find(consumer->id) == visited.end()) {
        if (dynamic_cast<SubgraphOp *>(consumer)) {
          auto subindex = consumer->input->indicesMap().at(tensor)[0];
          if (consumer->hasVirtualGraphId()) {
            // Also works if the callee is another subgraph
            auto vgId =
                consumer->getIntrospectionInVirtualGraphId(subindex, visited);
            if (vgId.first > -1)
              return vgId;
          }
        }
        if (IpuCopyOp *copyConsumer = dynamic_cast<IpuCopyOp *>(consumer)) {
          return {copyConsumer->getSourceIpu(tensor_id),
                  copyConsumer->settings.tileSet};
        }
      }
    }

    // Fallback 1: The tensor has no consumer inside the graph,
    // look after the graph
    if (tensor->consumers.getOps().empty() && tensor->isGraphOutput()) {
      OutIndex sgOutIndex = tensor->getGraphOutputIndex();
      OutIndex opOutIndex = subgraphOutToOpOutIndex(sgOutIndex);
      if (output->hasIndex(opOutIndex)) {
        auto vgId = output->tensor(opOutIndex)
                        ->getVirtualGraphIdAndTileSetUnsafe(visited);
        if (vgId.first > -1) {
          return vgId;
        }
      }
    }

    // Fallback 2: The tensor knows it's own VGID
    // We ask this only after callee introspection, because otherwise the
    // CallOp's VGID will be reported, which can be wrong if it's nested
    // consuming operator is on another virtual graph.
    if (tensor->hasVirtualGraphId()) {
      // Tensor has VirtualGraphID given by it's producer or consumer
      auto vgId = tensor->getVirtualGraphIdAndTileSet(visited);
      if (vgId.first > -1) {
        return vgId;
      }
    }
  }

  // Fallback 3: No VGID determined by introspection or tensor
  return Op::hasVirtualGraphId()
             ? VGraphIdAndTileSet(Op::getVirtualGraphId(),
                                  getSettings().tileSet)
             : VGraphIdAndTileSet(unusedVGraphId, TileSet::Compute);
}

VGraphIdAndTileSet
SubgraphOp::getIntrospectionOutVirtualGraphId(OutIndex index,
                                              std::set<OpId> visited) const {
  visited.insert(id);

  InIndex subgraphOutIndex = opOutToSubgraphOutIndex(index);

  if (subgraphOutIndex > -1) {
    auto num_ids = getCalledGraph().getOutputIds().size();
    if (subgraphOutIndex >= num_ids) {
      throw error("[getIntrospectionOutVirtualGraphId] "
                  "SubgraphOp ({}) has {} subgraph inputs, "
                  "but requested index is {}",
                  debugName(),
                  num_ids,
                  subgraphOutIndex);
    }

    auto tensor_id = getCalledGraph().getOutputId(subgraphOutIndex);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // Callee introspection
    if (tensor->hasProducer()) {
      auto producer = tensor->getProducer();
      if (visited.find(producer->id) == visited.end()) {
        if (dynamic_cast<SubgraphOp *>(producer)) {
          auto subindex = producer->output->indicesMap().at(tensor)[0];
          if (producer->hasVirtualGraphId()) {
            // Also works if the callee is another subgraph
            auto vgId =
                producer->getIntrospectionOutVirtualGraphId(subindex, visited);
            if (vgId.first > -1) {
              return vgId;
            }
          }
        }
      }
    }

    // Fallback 1: The tensor has no producer inside the graph,
    // look before the graph
    if (tensor->isGraphInput()) {
      OutIndex sgInIndex = tensor->getGraphInputIndex();
      OutIndex opInIndex = subgraphInToOpInIndex(sgInIndex);
      auto vgId =
          output->tensor(opInIndex)->getVirtualGraphIdAndTileSetUnsafe(visited);
      if (vgId.first > -1) {
        return vgId;
      }
    }

    // Fallback 2: The tensor knows it's own VGID
    // We ask this only after callee introspection, because otherwise the
    // CallOp's VGID will be reported, which can be wrong if it's nested
    // consuming operator is on another virtual graph.
    if (tensor->hasVirtualGraphId()) {
      // Tensor has VirtualGraphID given by it's producer or consumer
      auto vgId = tensor->getVirtualGraphIdAndTileSet(visited);
      if (vgId.first > -1) {
        return vgId;
      }
    }
  }

  // Fallback 3: No VGID determined by introspection or tensor
  return Op::hasVirtualGraphId()
             ? VGraphIdAndTileSet(Op::getVirtualGraphId(),
                                  getSettings().tileSet)
             : VGraphIdAndTileSet(unusedVGraphId, TileSet::Compute);
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
                                          InIndex inIndex) {
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
                                          InIndex inIndex) {
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
                                             OutIndex outIndex) {
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
                                             OutIndex outIndex) {
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

} // namespace popart
