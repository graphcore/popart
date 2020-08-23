// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnx/onnx_pb.h>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/call.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/scope.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

CallOp::CallOp(const OperatorIdentifier &opid_, Graph &parent_, Graph &callee_)
    : SubgraphOp(opid_, {parent_, "", parent_.getScope()}), callee(callee_) {
  settings.name = logging::format("Call_{}", callee_.id);
}

void CallOp::setup() {
  // Assume output tensors are ordered the same as those
  // in the callee subgraph
  for (int i = 0; i < callee.get().getOutputIds().size(); i++) {
    TensorId calleeOutputId = callee.get().getOutputId(i);
    outInfo(i) = callee.get().getTensors().get(calleeOutputId)->info;
  }
}

std::unique_ptr<Op> CallOp::clone() const {
  return std::make_unique<CallOp>(*this);
}

Graph &CallOp::getCalledGraph() const { return callee.get(); }

void CallOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("callee", callee.get().id.str());
}

view::Regions CallOp::modifies(InIndex in) const {
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

view::Regions CallOp::aliases(InIndex in, OutIndex out) const {
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

std::vector<const Graph *> CallOp::getCalledGraphs() const {
  return {&getCalledGraph()};
}

std::vector<TensorId> CallOp::getInputsForGraph(const Graph &) const {
  std::vector<TensorId> result;
  for (int i = 0; i < input->n(); i++) {
    result.push_back(inId(i));
  }
  return result;
}

VGraphIdAndIoTile
CallOp::getIntrospectionInVirtualGraphId(InIndex index) const {
  if (index > -1) {
    auto num_ids = getCalledGraph().getInputIds().size();
    if (index >= num_ids)
      throw error("[getIntrospectionInVirtualGraphId] "
                  "CallOp ({}) has {} inputs, but requested index is {}",
                  debugName(),
                  num_ids,
                  index);

    auto tensor_id = getCalledGraph().getInputId(index);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // Callee introspection
    for (auto consumer : tensor->consumers.getOps()) {
      if (dynamic_cast<CallOp *>(consumer)) {
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
      auto vgId = tensor->getVirtualGraphIdAndIoTile();
      if (vgId.first > -1) {
        return vgId;
      }
    }
  }

  // Fallback 2: No VGID determined by introspection or tensor
  return Op::hasVirtualGraphId()
             ? VGraphIdAndIoTile(Op::getVirtualGraphId(), getSettings().tileSet)
             : VGraphIdAndIoTile(unusedVGraphId, TileSet::Compute);
}

VGraphIdAndIoTile
CallOp::getIntrospectionOutVirtualGraphId(OutIndex index) const {
  if (index > -1) {
    auto num_ids = getCalledGraph().getOutputIds().size();
    if (index >= num_ids)
      throw error("[getIntrospectionOutVirtualGraphId] "
                  "CallOp ({}) has {} inputs, but requested index is {}",
                  debugName(),
                  num_ids,
                  index);

    auto tensor_id = getCalledGraph().getOutputId(index);
    auto tensor    = getCalledGraph().getTensors().get(tensor_id);

    // Callee introspection
    auto producer = tensor->getProducer();
    if (dynamic_cast<CallOp *>(producer)) {
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
      auto vgId = tensor->getVirtualGraphIdAndIoTile();
      if (vgId.first > -1) {
        return vgId;
      }
    }
  }

  // Fallback 2: No VGID determined by introspection or tensor
  return Op::hasVirtualGraphId()
             ? VGraphIdAndIoTile(Op::getVirtualGraphId(), getSettings().tileSet)
             : VGraphIdAndIoTile(unusedVGraphId, TileSet::Compute);
}

void CallOp::addAlias(InIndex in,
                      OutIndex out,
                      view::Chains fwdChains,
                      view::Chains bwdChains) {
  aliasMap.insert(std::make_pair(std::make_pair(in, out),
                                 std::make_pair(fwdChains, bwdChains)));
}

void CallOp::addModified(InIndex in, view::Regions regions) {
  modifiesMap.insert(std::make_pair(in, regions));
}

view::RegMap CallOp::fwdRegMap(InIndex inIndex, OutIndex outIndex) const {
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

view::RegMap CallOp::bwdRegMap(InIndex inIndex, OutIndex outIndex) const {
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

GraphId CallOp::getBackwardsGraphId() const {
  return GraphId(logging::format("{}_bwd", callee.get().id));
}

std::vector<std::unique_ptr<Op>> CallOp::getGradOps() {
  auto gradInInfo =
      getCalledGraph().getBackwardsGraph(getBackwardsGraphId()).gradInputInfo();

  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<CallGradOp>(*this, gradInInfo));
  return upops;
}

CallGradOp::CallGradOp(CallOp &fwdOp,
                       const std::vector<GradInOutMapper> &gradInInfo_)
    : CallOp(Onnx::CustomOperators::Call_1,
             fwdOp.getGraph(),
             fwdOp.getCalledGraph().getBackwardsGraph(
                 fwdOp.getBackwardsGraphId())),
      gradInInfo(gradInInfo_) {
  // An output for every input to the forward CallOp
  for (int i = 0; i < fwdOp.input->n(); i++) {
    outInfoMap.insert({i, i});
  }
}

const std::vector<GradInOutMapper> &CallGradOp::gradInputInfo() const {
  return gradInInfo;
}

const std::map<int, int> &CallGradOp::gradOutToNonGradIn() const {
  return outInfoMap;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};

static OpDefinition callOpDef({OpDefinition::Inputs({{"inputs", T}}),
                               OpDefinition::Outputs({{"outputs", T}}),
                               OpDefinition::Attributes({{"callee", {"*"}}})});

static OpCreator<CallOp> callOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Call_1, callOpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
      ONNX_NAMESPACE::GraphProto callee =
          info.attributes.getAttribute<Attributes::Graph>("callee");

      if (callee.name().empty()) {
        throw error("CallOp subgraph must be named, so that it can be "
                    "identified for re-use.");
      }

      // If the callee subgraph has already been constructed, get that.
      // Otherwise, construct here.
      auto &ir = info.settings.graph.get().getIr();
      Graph *calleeGraph;
      if (ir.hasGraph(callee.name())) {
        calleeGraph = &ir.getGraph(callee.name());
      } else {
        calleeGraph = &ir.createGraph(callee.name());

        // Find the input tensors in the parent graph (or its called
        // graphs) to determine the tensor type
        auto inputs = info.getInputIds();
        std::map<TensorId, TensorInfo> inputInfos;
        for (auto &graph : ir.getAllGraphs()) {
          for (TensorId input : inputs) {
            if (graph->getTensors().contains(input, graph->getScope())) {
              TensorId tid = graph->getTensors().find(input, graph->getScope());
              Tensor *tensor = graph->getTensors().get(tid);
              inputInfos.emplace(input, tensor->info);
            }
          }
        }

        // Check that an InputInfo was found for all inputs
        for (TensorId input : inputs) {
          if (inputInfos.count(input) == 0) {
            throw error(
                "Unable to determine tensor info for input to CallOp, {}",
                input);
          }
        }

        for (int i = 0; i < callee.input_size(); i++) {
          // Assume callee graph inputs are in the same order as this
          // op's node inputs
          TensorInfo calleeInputInfo = inputInfos.at(inputs.at(i));
          auto scopedId = calleeGraph->addScope(callee.input(i).name());
          calleeGraph->addInput(scopedId, calleeInputInfo);
        }

        calleeGraph->constructFromOnnxGraph(callee);

        for (auto &output : callee.output()) {
          auto scopedId = calleeGraph->addScope(output.name());
          calleeGraph->markAsOutput(scopedId);
        }
      }

      return std::unique_ptr<Op>(
          new CallOp(info.opid, info.settings.graph.get(), *calleeGraph));
    },
    true);
} // namespace

} // namespace popart
