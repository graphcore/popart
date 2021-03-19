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

LoopOp::LoopOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               Graph &callee_)
    : SubgraphOp(_opid, settings_), callee(callee_), tripCountValue(0),
      numImplicitScanOutputs(0) {}

LoopOp::LoopOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               Graph &callee_,
               std::vector<std::pair<TensorId, TensorInfo>> opInputs_,
               std::vector<TensorId> implicitTensors_,
               int numImplicitScanOutputs_)
    : SubgraphOp(_opid, settings_), callee(callee_), tripCountValue(0),
      numImplicitScanOutputs(numImplicitScanOutputs_) {
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
  Op::appendOutlineAttributes(os);
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

std::vector<const Graph *> LoopOp::getCalledGraphs() const {
  return {&getCalledGraph()};
}

void LoopOp::connectInTensor(InIndex inIndex, TensorId tensorId) {
  if (inIndex == LoopOp::getMaximumTripCountInIndex()) {
    logging::op::info(
        "INT64 is currently not supported. Casting loop input {} to INT32",
        tensorId);
    Tensor *tensor     = getGraph().getTensors().get(tensorId);
    int64_t tensorData = *static_cast<int64_t *>(tensor->tensorData()->data());

    std::vector<int32_t> castTensorData{static_cast<int32_t>(tensorData)};
    tripCountValue        = castTensorData.front();
    TensorId castTensorId = getIr().createIntermediateTensorId(tensorId);

    getGraph().getTensors().addConstInit(castTensorId,
                                         {DataType::INT32, {}},
                                         castTensorData.data(),
                                         tensor->getDebugInfo());
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
      auto &parentGraph = info.settings.graph.get();
      auto &tensors     = parentGraph.getTensors();

      auto loopBodyInputs  = SubgraphOp::getBodyInputIds(callee);
      auto loopBodyOutputs = SubgraphOp::getBodyOutputIds(callee);

      std::vector<std::pair<TensorId, TensorInfo>> opInputs;

      for (int i = 0; i < info.getInputIds().size(); ++i) {
        opInputs.push_back({info.getInputIds().at(i),
                            tensors.get(info.getInputIds().at(i))->info});
      }

      auto implicitTensors =
          SubgraphOp::getImplicitTensors(callee, tensors, opInputs);

      logging::op::trace("[LoopOp] Implicit tensors: {}", implicitTensors);

      auto subgraphId =
          callee.name().empty()
              ? parentGraph.getIr().createUniqueSubgraphId({"loop"})
              : callee.name();
      auto &ir          = parentGraph.getIr();
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

      int numImplicitScanOutputs =
          calleeGraph.getOutputIds().size() - loopBodyInputs.size() + 1;

      return std::unique_ptr<Op>(new LoopOp(info.opid,
                                            info.settings,
                                            calleeGraph,
                                            opInputs,
                                            implicitTensors,
                                            numImplicitScanOutputs));
    },
    true);

} // namespace

} // namespace popart
