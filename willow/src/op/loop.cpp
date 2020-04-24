// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <onnx/onnx_pb.h>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/loop.hpp>
#include <popart/opmanager.hpp>
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

std::vector<std::pair<int, TensorId>>
pairInputIdxToInputId(const ONNX_NAMESPACE::GraphProto &bodyProto) {
  std::vector<std::pair<int, TensorId>> bodyInputs;
  for (int i = 0; i < bodyProto.input_size(); ++i) {
    bodyInputs.push_back(std::make_pair(i, bodyProto.input(i).name()));
  }

  return bodyInputs;
}

std::vector<std::pair<int, TensorId>>
pairOutputIdxToOutputId(const ONNX_NAMESPACE::GraphProto &body) {
  std::vector<std::pair<int, TensorId>> bodyOutputs;
  for (int i = 0; i < body.output_size(); ++i) {
    bodyOutputs.push_back(std::make_pair(i, body.output(i).name()));
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

LoopOp::LoopOp(
    const OperatorIdentifier &opid,
    const Op::Settings &settings,
    const GraphId &subgraphId,
    const std::vector<std::pair<TensorId, TensorInfo>> inputs,
    const std::vector<TensorId> implicitTensors,
    const std::vector<std::pair<TensorId, TensorInfo>> explicitTensors)
    : Op(opid, settings), tripCountValue_(0), subgraphId_(subgraphId),
      implicitTensors_(implicitTensors), inputs_(inputs),
      explicitTensors_(explicitTensors) {}

void LoopOp::setup() {

  // Mark implicit inputs to the LoopOp as implicit
  for (auto &tid : implicitTensors_) {
    auto tensor = getGraph().getTensors().get(tid);
    tensor->setImplicitLoopInput(true);
  }

  for (int i = input->n(); i < inputs_.size(); ++i) {
    connectInTensor(i, inputs_.at(i).first);
  }

  // Connect the output of the subgraph with the output from the main graph
  // an offset of +1 because the boolean termination is the first variable
  for (int i = 0; i < output->n(); ++i) {
    auto tensorId = subgraph().getOutputId(i + 1);
    auto tensor   = subgraph().getTensors().get(tensorId);
    outInfo(i)    = tensor->info;
  }
}

std::unique_ptr<Op> LoopOp::clone() const {
  return std::make_unique<LoopOp>(*this);
}

void LoopOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
}

Graph &LoopOp::subgraph() const {
  return getGraph().getIr().getGraph(subgraphId_);
}

std::vector<const Graph *> LoopOp::getCalledGraphs() const {
  return {&subgraph()};
}

std::vector<TensorId> LoopOp::getInputsForGraph(const Graph &graph) const {
  auto allTensors = graph.getIr().getMainGraphTensors().getAllTensorIds();
  std::vector<TensorId> inputIds;
  for (int i = 0; i < input->n(); ++i) {
    auto found =
        std::find(std::begin(allTensors), std::end(allTensors), inId(i));
    if (found != std::end(allTensors)) {
      inputIds.push_back(inId(i));
    }
  }
  return inputIds;
}

void LoopOp::connectInTensor(InIndex inIndex, TensorId tensorId) {
  if (inIndex == 0) {
    logging::op::warn(
        "INT64 is currently not supported. Casting loop input {} to INT32",
        tensorId);
    Tensor *tensor     = getGraph().getTensors().get(tensorId);
    int64_t tensorData = *static_cast<int64_t *>(tensor->tensorData()->data());

    std::vector<int32_t> castTensorData{static_cast<int32_t>(tensorData)};
    tripCountValue_         = castTensorData.front();
    TensorId castTensorName = tensorId + "_int32";

    getGraph().getTensors().addConstInit(
        castTensorName, {DataType::INT32, {}}, castTensorName.data());
    defaultConnectInTensor(inIndex, castTensorName);
  } else {
    defaultConnectInTensor(inIndex, tensorId);
  };
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
      const ONNX_NAMESPACE::GraphProto &bodyGraph =
          info.attributes.getAttribute<Attributes::Graph>("body");
      auto &mainGraph = info.settings.graph.get();
      auto &tensors   = mainGraph.getTensors();

      auto loopBodyInputs  = pairInputIdxToInputId(bodyGraph);
      auto loopBodyOutputs = pairOutputIdxToOutputId(bodyGraph);

      std::vector<std::pair<TensorId, TensorInfo>> opInputs, explicitTensors;

      for (int i = 0; i < info.getInputIds().size(); ++i) {
        explicitTensors.push_back(
            std::make_pair(loopBodyInputs[i].second,
                           tensors.get(info.getInputIds().at(i))->info));
        opInputs.push_back(
            std::make_pair(loopBodyInputs[i].second,
                           tensors.get(info.getInputIds().at(i))->info));
      }

      auto implicitTensors = addImplicitTensors(bodyGraph, tensors, opInputs);

      auto subgraphId = bodyGraph.name().empty() ? generateSubgraphUID("loop")
                                                 : bodyGraph.name();
      auto &ir       = mainGraph.getIr();
      auto &subgraph = ir.createGraph(subgraphId);

      for (int i = 0; i < opInputs.size(); ++i) {
        auto &kv = opInputs.at(i);
        if (i == 0) {
          TensorInfo tensorInfo = {DataType::INT32, kv.second.shape()};
          TensorId scope        = subgraph.addScope(kv.first);
          subgraph.addInput(scope, tensorInfo);
        } else {
          TensorId scope = subgraph.addScope(kv.first);
          subgraph.addInput(scope, kv.second);
        }
      }

      subgraph.constructFromOnnxGraph(bodyGraph);

      for (const auto &kv : loopBodyOutputs) {
        TensorId scope = subgraph.addScope(kv.second);
        subgraph.markAsOutput(scope);
      }

      return std::unique_ptr<Op>(new LoopOp(info.opid,
                                            info.settings,
                                            subgraphId,
                                            opInputs,
                                            implicitTensors,
                                            explicitTensors));
    },
    true);

} // namespace

} // namespace popart
