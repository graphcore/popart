// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>
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

namespace popart {

namespace {
GraphId generateSubgraphUID(const std::string &postfix) {
  static int uid = 0;
  return GraphId(logging::format("loopop_subgraph_{}_{}", uid++, postfix));
}

bool existsInLoopBodyInputs(std::vector<std::string> &loopBodyInputIds,
                            TensorId &tensorId) {
  auto found = std::find(
      std::begin(loopBodyInputIds), std::end(loopBodyInputIds), tensorId);
  if (found != std::end(loopBodyInputIds)) {
    return true;
  }
  return false;
}

bool existsInOpInputs(std::vector<std::pair<TensorId, TensorInfo>> &opInputs,
                      TensorId &tensorId) {
  auto found =
      std::find_if(std::begin(opInputs),
                   std::end(opInputs),
                   [&tensorId](const std::pair<TensorId, TensorInfo> &kv) {
                     return kv.first == tensorId;
                   });
  if (found != std::end(opInputs)) {
    return true;
  }
  return false;
}

std::vector<TensorId>
getBodyInputIds(const ONNX_NAMESPACE::GraphProto &bodyProto) {
  std::vector<TensorId> bodyInputIds;
  for (const auto &input : bodyProto.input()) {
    bodyInputIds.push_back(input.name());
  }
  return bodyInputIds;
}

std::vector<TensorId>
loopBodyInputIds(const ONNX_NAMESPACE::GraphProto &bodyProto) {
  std::vector<TensorId> bodyInputs;
  for (int i = 0; i < bodyProto.input_size(); ++i) {
    bodyInputs.push_back(bodyProto.input(i).name());
  }
  return bodyInputs;
}

std::vector<TensorId>
loopBodyOutputIds(const ONNX_NAMESPACE::GraphProto &bodyProto) {
  std::vector<TensorId> bodyOutputs;
  for (int i = 0; i < bodyProto.output_size(); ++i) {
    bodyOutputs.push_back(bodyProto.output(i).name());
  }
  return bodyOutputs;
}

std::vector<TensorId>
addImplicitTensors(const ONNX_NAMESPACE::GraphProto &bodyProto,
                   popart::Tensors &tensors,
                   std::vector<std::pair<TensorId, TensorInfo>> &allOpInputs) {

  auto loopBodyInputIds = getBodyInputIds(bodyProto);
  std::vector<TensorId> implicitTensors;

  for (int i = 0; i < bodyProto.node_size(); ++i) {
    auto &nodeProto = bodyProto.node(i);
    for (int j = 0; j < nodeProto.input_size(); ++j) {
      auto tid        = nodeProto.input(j);
      auto inLoopBody = existsInLoopBodyInputs(loopBodyInputIds, tid);
      if (!inLoopBody) {
        auto inOpInputs = existsInOpInputs(allOpInputs, tid);
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

} // namespace

LoopOp::LoopOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               Graph &callee_)
    : SubgraphOp(_opid, settings_), callee(callee_), tripCountValue(0) {}

LoopOp::LoopOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               Graph &callee_,
               std::vector<std::pair<TensorId, TensorInfo>> opInputs_,
               std::vector<TensorId> implicitTensors_)
    : SubgraphOp(_opid, settings_), callee(callee_), tripCountValue(0) {
  for (int i = 0; i < opInputs_.size(); ++i) {
    TensorId inId = opInputs_.at(i).first;
    if (std::find(implicitTensors_.begin(), implicitTensors_.end(), inId) !=
        implicitTensors_.end()) {
      connectInTensor(i, opInputs_.at(i).first);
    }
  }
}

void LoopOp::setup() {
  // Connect the output of the subgraph with the output from the main graph
  // an offset of +1 because the boolean termination is the first variable
  // Skip the cond-out tensor
  // Body out 0   ->  skip
  // Body out 1   ->  Loop out 0
  // ..
  // Body out M-1 ->  Loop out M-2
  for (int i = 0; i < output->n(); ++i) {
    auto tensorId = getCalledGraph().getOutputId(i + 1);
    auto tensor   = getCalledGraph().getTensors().get(tensorId);
    outInfo(i)    = tensor->info;
  }
}

std::unique_ptr<Op> LoopOp::clone() const {
  return std::make_unique<LoopOp>(*this);
}

void LoopOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("callee", callee.get().id.str());
  os.appendAttribute("tripCountValue", tripCountValue);
}

Graph &LoopOp::getCalledGraph() const { return callee.get(); }

int LoopOp::numExplicitInputs() const {
  int numOutputs = getCalledGraph().getOutputIds().size() - 1;
  // User defined explicit inputs + trip count and termination condition
  int numExplicitInputs = numOutputs + 2;
  return numExplicitInputs;
}

int LoopOp::numImplicitInputs() const {
  return input->maxIndex() + 1 - numExplicitInputs();
}

InIndex LoopOp::subgraphInToOpInIndex(InIndex index) const { return index; }

InIndex LoopOp::opInToSubgraphInIndex(InIndex index) const { return index; }

OutIndex LoopOp::subgraphOutToOpOutIndex(OutIndex index) const {
  return index - 1;
}
OutIndex LoopOp::opOutToSubgraphOutIndex(OutIndex index) const {
  return index + 1;
}

std::vector<const Graph *> LoopOp::getCalledGraphs() const {
  return {&getCalledGraph()};
}

void LoopOp::connectInTensor(InIndex inIndex, TensorId tensorId) {
  if (inIndex == LoopOp::getMaximumTripCountInIndex()) {
    logging::op::warn(
        "INT64 is currently not supported. Casting loop input {} to INT32",
        tensorId);
    Tensor *tensor     = getGraph().getTensors().get(tensorId);
    int64_t tensorData = *static_cast<int64_t *>(tensor->tensorData()->data());

    std::vector<int32_t> castTensorData{static_cast<int32_t>(tensorData)};
    tripCountValue        = castTensorData.front();
    TensorId castTensorId = getIr().createIntermediateTensorId(tensorId);

    getGraph().getTensors().addConstInit(
        castTensorId, {DataType::INT32, {}}, castTensorData.data());
    defaultConnectInTensor(inIndex, castTensorId);
  } else {
    defaultConnectInTensor(inIndex, tensorId);
  };
}

void LoopOp::addLoopInput(InIndex index,
                          TensorId tensorId,
                          TensorId subgraphTensorId,
                          bool overwrite) {
  if (!overwrite) {
    int n = input->maxIndex();
    for (InIndex i = n; i >= index; --i) {
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
  int n = input->maxIndex();
  for (InIndex i = index; i <= n; ++i) {
    if (hasInput(i + 1)) {
      connectInTensor(i, input->tensorIdMap().at(i + 1));
      disconnectInTensor(i + 1);
    }
  }
  getCalledGraph().removeInput(opInToSubgraphInIndex(index));
}

void LoopOp::removeLoopOutput(OutIndex index) {
  disconnectOutTensor(output->tensor(index));
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
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
      const ONNX_NAMESPACE::GraphProto &callee =
          info.attributes.getAttribute<Attributes::Graph>("body");
      auto &mainGraph = info.settings.graph.get();
      auto &tensors   = mainGraph.getTensors();

      auto loopBodyInputs  = loopBodyInputIds(callee);
      auto loopBodyOutputs = loopBodyOutputIds(callee);

      std::vector<std::pair<TensorId, TensorInfo>> opInputs;

      for (int i = 0; i < info.getInputIds().size(); ++i) {
        opInputs.push_back({info.getInputIds().at(i),
                            tensors.get(info.getInputIds().at(i))->info});
      }

      auto implicitTensors = addImplicitTensors(callee, tensors, opInputs);

      auto subgraphId =
          callee.name().empty() ? generateSubgraphUID("loop") : callee.name();
      auto &ir          = mainGraph.getIr();
      auto &calleeGraph = ir.createGraph(subgraphId);

      for (int i = 0; i < opInputs.size(); ++i) {
        auto &kv = opInputs.at(i);
        TensorId scopedTensorId;
        if (i < loopBodyInputs.size()) {
          // Explicit
          scopedTensorId = calleeGraph.addScope(loopBodyInputs.at(i));
        } else {
          // Implicit
          scopedTensorId = calleeGraph.addScope(kv.first);
        }
        if (i == 0) {
          TensorInfo tensorInfo = {DataType::INT32, kv.second.shape()};
          calleeGraph.addInput(scopedTensorId, tensorInfo);
        } else {
          calleeGraph.addInput(scopedTensorId, kv.second);
        }
      }

      calleeGraph.constructFromOnnxGraph(callee);

      for (TensorId outputId : loopBodyOutputs) {
        TensorId scopedTensorId = calleeGraph.addScope(outputId);
        calleeGraph.markAsOutput(scopedTensorId);
      }

      return std::unique_ptr<Op>(new LoopOp(
          info.opid, info.settings, calleeGraph, opInputs, implicitTensors));
    },
    true);

} // namespace

} // namespace popart
