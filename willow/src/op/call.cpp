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

std::vector<const Graph *> CallOp::getCalledGraphs() const {
  return {&getCalledGraph()};
}

void CallOp::setCalledGraph(Graph &graph) { callee = graph; }

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
