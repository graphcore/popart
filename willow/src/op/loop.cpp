// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>

namespace popart {

LoopOp::LoopOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               Graph &callee_)
    : SubgraphOp(_opid, settings_), callee(callee_), tripCountValue(0),
      numImplicitScanOutputs(0) {}

LoopOp::LoopOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               Graph &callee_,
               int numImplicitScanOutputs_)
    : SubgraphOp(_opid, settings_), callee(callee_), tripCountValue(0),
      numImplicitScanOutputs(numImplicitScanOutputs_) {}

void LoopOp::setup() {
  // Connect the output of the subgraph with the output from the main graph
  // an offset of +1 because the boolean termination is the first variable
  // Skip the cond-out tensor
  // Body out 0   ->  skip
  // Body out 1   ->  Loop out 0
  // ..
  // Body out M-1 ->  Loop out M-2
  // Body out M   ->  Loop out M-1 (implicit scan output)
  // ..
  // Body out K   ->  Loop out K-1 (implicit scan output)
  for (int i = 0; i < output->n(); ++i) {
    auto tensorId = getCalledGraph().getOutputId(i + 1);
    auto tensor   = getCalledGraph().getTensors().get(tensorId);
    outInfo(i)    = tensor->info;
    // Implicit scan output
    if (i >= output->n() - getNumImplicitScanOutputs()) {
      auto shape = outInfo(i).shape();
      shape.insert(shape.begin(), static_cast<int64_t>(getTripCountValue()));
      outInfo(i).set(outInfo(i).dataType(), shape, outInfo(i).metaShape());
    }
  }
}

std::unique_ptr<Op> LoopOp::clone() const {
  return std::make_unique<LoopOp>(*this);
}

void LoopOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  SubgraphOp::appendOutlineAttributes(os);
  os.appendAttribute("callee", callee.get().id.str());
  os.appendAttribute("tripCountValue", tripCountValue);
}

Graph &LoopOp::getCalledGraph() const { return callee.get(); }

void LoopOp::setCalledGraph(Graph &graph) { callee = graph; }

int LoopOp::getNumExplicitInputs() const {
  int numOutputs =
      getCalledGraph().getOutputIds().size() - 1 - numImplicitScanOutputs;
  // User defined explicit inputs + trip count and termination condition
  int numExplicitInputs = numOutputs + 2;
  return numExplicitInputs;
}

int LoopOp::getNumImplicitInputs() const {
  return input->maxIndex() + 1 - getNumExplicitInputs();
}

InIndex LoopOp::subgraphInToOpInIndex(InIndex index) const { return index; }

InIndex LoopOp::opInToSubgraphInIndex(InIndex index) const { return index; }

OutIndex LoopOp::subgraphOutToOpOutIndex(OutIndex index) const {
  return index - 1;
}
OutIndex LoopOp::opOutToSubgraphOutIndex(OutIndex index) const {
  return index + 1;
}

VGraphIdAndTileSet
LoopOp::getIntrospectionInVirtualGraphId(InIndex index,
                                         std::set<OpId> &visited) const {
  auto vgid = SubgraphOp::getIntrospectionInVirtualGraphId(index, visited);
  if (settings.vgraphId && vgid.first == unusedVGraphId &&
      (index == LoopOp::getMaximumTripCountInIndex() ||
       index == LoopOp::getTerminationConditionInIndex())) {
    return {*(settings.vgraphId), settings.tileSet};
  }
  return vgid;
}

VGraphIdAndTileSet
LoopOp::getIntrospectionOutVirtualGraphId(OutIndex index,
                                          std::set<OpId> &visited) const {
  return SubgraphOp::getIntrospectionOutVirtualGraphId(index, visited);
}

std::vector<const Graph *> LoopOp::getCalledGraphs() const {
  return {&getCalledGraph()};
}

void LoopOp::connectInTensor(InIndex inIndex, TensorId tensorId) {
  defaultConnectInTensor(inIndex, tensorId);

  if (inIndex == LoopOp::getMaximumTripCountInIndex()) {
    Tensor *tensor = getGraph().getTensors().get(tensorId);
    switch (tensor->info.dataType()) {
    case DataType::INT32: {
      int32_t tensorData = *reinterpret_cast<int32_t *>(
          tensor->getDataViaGraphTraversal().data());
      std::vector<int32_t> castTensorData{static_cast<int32_t>(tensorData)};
      tripCountValue = castTensorData.front();
      break;
    }
    case DataType::UINT32: {
      uint32_t tensorData = *reinterpret_cast<uint32_t *>(
          tensor->getDataViaGraphTraversal().data());
      std::vector<int32_t> castTensorData{static_cast<int32_t>(tensorData)};
      tripCountValue = castTensorData.front();
      break;
    }
    case DataType::INT64: {
      int64_t tensorData = *reinterpret_cast<int64_t *>(
          tensor->getDataViaGraphTraversal().data());
      std::vector<int32_t> castTensorData{static_cast<int32_t>(tensorData)};
      tripCountValue = castTensorData.front();
      break;
    }
    case DataType::UINT64: {
      uint64_t tensorData = *reinterpret_cast<uint64_t *>(
          tensor->getDataViaGraphTraversal().data());
      std::vector<int32_t> castTensorData{static_cast<int32_t>(tensorData)};
      tripCountValue = castTensorData.front();
      break;
    }
    default:
      throw error("[LoopOp] unsupported trip count data type {}",
                  tensor->info.dataType());
    }
    logging::op::debug("[LoopOp] {} updated trip count to {} (from {})",
                       debugName(),
                       tripCountValue,
                       tensor->id);
  }
}

void LoopOp::addLoopInput(InIndex index,
                          TensorId tensorId,
                          TensorId subgraphTensorId,
                          bool overwrite) {
  if (!overwrite) {
    int n = input->maxIndex();
    for (InIndex i = n; i >= index; --i) {
      adjustModifiedIndices(i, i + 1);
      adjustAliasInIndices(i, i + 1);
      if (hasInput(i + 1)) {
        disconnectInTensor(i + 1);
      }
      connectInTensor(i + 1, input->tensorIdMap().at(i));
    }
  }
  if (hasInput(index)) {
    disconnectInTensor(index);
  }
  connectInTensor(index, tensorId);
  getCalledGraph().addInput(opInToSubgraphInIndex(index),
                            subgraphTensorId,
                            getIr().getTensor(tensorId)->info,
                            overwrite);
}

void LoopOp::addLoopOutput(OutIndex index,
                           TensorId tensorId,
                           TensorId subgraphTensorId,
                           bool overwrite) {
  if (!overwrite) {
    int n = output->maxIndex();
    for (OutIndex i = n; i >= index; --i) {
      if (output->hasIndex(i)) {
        adjustAliasOutIndices(i, i + 1);
        Tensor *t = output->tensorMap().at(i);
        disconnectOutTensor(t);
        connectOutTensor(i + 1, t->id);
      }
    }
  }
  if (output->hasIndex(index)) {
    Tensor *t = output->tensorMap().at(index);
    disconnectOutTensor(t);
  }
  if (getIr().containsTensor(tensorId)) {
    Tensor *t = getIr().getTensor(tensorId);
    if (t->hasProducer()) {
      t->getProducer()->disconnectOutTensor(t);
    }
    connectOutTensor(index, tensorId);
  } else {
    createAndConnectOutTensor(index, tensorId);
  }
  getCalledGraph().markAsOutput(
      opOutToSubgraphOutIndex(index), subgraphTensorId, overwrite);
}

void LoopOp::removeLoopInput(InIndex index) {
  disconnectInTensor(index);
  removeModified(index);
  for (auto &out : output->tensorMap()) {
    removeAlias(index, out.first);
  }
  int n = input->maxIndex();
  for (InIndex i = index; i <= n; ++i) {
    if (hasInput(i + 1)) {
      connectInTensor(i, input->tensorIdMap().at(i + 1));
      disconnectInTensor(i + 1);
      adjustModifiedIndices(i + 1, i);
      adjustAliasInIndices(i + 1, i);
    }
  }
  getCalledGraph().removeInput(opInToSubgraphInIndex(index));
}

void LoopOp::removeLoopOutput(OutIndex index) {
  disconnectOutTensor(output->tensor(index));
  for (auto &in : input->tensorMap()) {
    removeAlias(in.first, index);
  }
  int n = output->maxIndex();
  for (InIndex i = index; i <= n; ++i) {
    if (output->hasIndex(i + 1)) {
      Tensor *t = output->tensor(i + 1);
      disconnectOutTensor(t);
      connectOutTensor(i, t->id);
    }
  }
  getCalledGraph().removeOutput(opOutToSubgraphOutIndex(index));
}

std::set<OutIndex> LoopOp::opInToOpOutIndex(InIndex in) const {
  std::set<OutIndex> indices;
  auto outIndex = in - getFirstInputInIndex() + getFirstOutputOutIndex();
  if (hasOutput(outIndex)) {
    indices.insert(outIndex);
  }
  return indices;
}

std::set<InIndex> LoopOp::opOutToOpInIndex(OutIndex out) const {
  std::set<OutIndex> indices;
  auto inIndex = out + getFirstInputInIndex() - getFirstOutputOutIndex();
  if (hasInput(inIndex)) {
    indices.insert(inIndex);
  }
  return indices;
}

namespace {

static OpDefinition::DataTypes I = {DataType::INT64};
static OpDefinition::DataTypes B = {DataType::BOOL};
static OpDefinition::DataTypes V = {DataType::BOOL,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition
    loopOpDef({OpDefinition::Inputs({{"M", I}, {"cond", V}, {"v_initial", V}}),
               OpDefinition::Outputs({{"outputs", V}}),
               OpDefinition::Attributes({
                   {"body", {"*"}},
               })});

static OpCreator<LoopOp> loopOpCreator(
    OpDefinitions({{Onnx::Operators::Loop_1, loopOpDef},
                   {Onnx::Operators::Loop_11, loopOpDef}}),
    [](const OpCreatorInfo &info, Graph &graph) -> Op * {
      const ONNX_NAMESPACE::GraphProto &callee =
          info.attributes.getAttribute<Attributes::Graph>("body");
      auto &parentGraph = info.settings.graph.get();
      auto &ir          = parentGraph.getIr();
      auto &tensors     = parentGraph.getTensors();

      auto loopBodyInputs  = SubgraphOp::getBodyInputIds(callee);
      auto loopBodyOutputs = SubgraphOp::getBodyOutputIds(callee);

      std::vector<std::pair<TensorId, TensorInfo>> opInputs;

      if (info.hasInputIds()) {
        for (int i = 0; i < info.getInputIds().size(); ++i) {
          logging::op::trace(
              "[LoopOp] Op input: {} - {}", i, info.getInputIds().at(i));
          opInputs.push_back({info.getInputIds().at(i),
                              tensors.get(info.getInputIds().at(i))->info});
        }
      }

      std::vector<TensorId> parentScopedImplicitTensorIds;
      auto implicitTensorIds = onnxutil::getImplicitTensorIds(callee);
      for (auto implicitTensorId : implicitTensorIds) {
        auto parentScopedImplicitTensorId =
            addScope(parentGraph, implicitTensorId);
        Tensor *tensor =
            parentGraph.getTensors().get(parentScopedImplicitTensorId);
        if (std::find(parentScopedImplicitTensorIds.begin(),
                      parentScopedImplicitTensorIds.end(),
                      parentScopedImplicitTensorId) ==
            parentScopedImplicitTensorIds.end()) {
          opInputs.push_back({implicitTensorId, tensor->info});
          parentScopedImplicitTensorIds.push_back(parentScopedImplicitTensorId);
        }
      }

      logging::op::trace("[LoopOp] Callee: {}, implicit tensors: {}",
                         callee.name(),
                         implicitTensorIds);

      GraphId subgraphId("");

      if (callee.name().empty()) {
        subgraphId = parentGraph.getIr().createUniqueSubgraphId({"loop"});
      } else {
        subgraphId = callee.name();
      }

      if (ir.hasGraph(subgraphId)) {
        subgraphId = parentGraph.getIr().createUniqueSubgraphId(subgraphId);
      }

      auto &calleeGraph = ir.createGraph(subgraphId);

      for (int i = 0; i < opInputs.size(); ++i) {
        auto &kv = opInputs.at(i);
        TensorId scopedTensorId;
        if (i < loopBodyInputs.size()) {
          // Explicit
          scopedTensorId = addScope(calleeGraph, loopBodyInputs.at(i));
        } else {
          // Implicit
          scopedTensorId = addScope(calleeGraph, kv.first);
        }
        logging::op::trace("[LoopOp] Callee: {}, input: {} - {} -> {}",
                           callee.name(),
                           i,
                           kv.first,
                           scopedTensorId);
        calleeGraph.addInput(scopedTensorId, kv.second);
      }

      int numImplicitScanOutputs =
          loopBodyOutputs.size() - loopBodyInputs.size() + 1;
      Op *op = graph.createOp<LoopOp>(
          info.opid, info.settings, calleeGraph, numImplicitScanOutputs);

      // Connect explicit inputs
      if (info.hasInputIds()) {
        for (InIndex i = 0; i < info.getInputIds().size(); ++i) {
          auto scopedName =
              graph.getTensors().find(info.getInputIds().at(i), op->getScope());
          op->connectInTensor(i, scopedName);
        }
      }

      // Connect implicit inputs
      for (auto parentScopedImplicitTensorId : parentScopedImplicitTensorIds) {
        op->connectInTensor(op->input->maxIndex() + 1,
                            parentScopedImplicitTensorId);
      }

      // Construct body graph
      calleeGraph.constructFromOnnxGraph(callee);

      // Mark body outputs
      for (TensorId outputId : loopBodyOutputs) {
        TensorId scopedTensorId = addScope(calleeGraph, outputId);
        calleeGraph.markAsOutput(scopedTensorId);
      }

      // Connect outputs
      if (info.hasOutputIds()) {
        for (OutIndex i = 0; i < info.getOutputIds().size(); ++i) {
          op->createAndConnectOutTensor(i, info.getOutputIds().at(i));
        }
      }

      logging::op::trace("[LoopOp] Callee: {}, inputs: {}, outputs: {}",
                         calleeGraph.id.str(),
                         calleeGraph.getInputIds(),
                         calleeGraph.getOutputIds());
      return op;
    },
    true);

} // namespace

} // namespace popart
