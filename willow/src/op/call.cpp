// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/call.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/scope.hpp>
#include <popart/tensorindex.hpp>
#include <popart/transforms/autodiff/calledgraphgradophelper.hpp>
#include <popart/util.hpp>

namespace popart {

CallOp::CallOp(const OperatorIdentifier &opid_,
               Graph &callee_,
               const Op::Settings &settings_)
    : CallOp(opid_, callee_, {}, settings_) {}

CallOp::CallOp(const OperatorIdentifier &opid_,
               Graph &callee_,
               const std::vector<int> &modifiedInputsViaAttrs_,
               const Op::Settings &settings_)
    : SubgraphOp(opid_, settings_), callee(callee_),
      modifiedInputsViaAttrs(modifiedInputsViaAttrs_) {}

void CallOp::setup() {
  // Assume output tensors are ordered the same as those
  // in the callee subgraph
  for (int i = 0; i < callee.get().getOutputIds().size(); i++) {
    TensorId calleeOutputId = callee.get().getOutputId(i);
    outInfo(i) = callee.get().getTensors().get(calleeOutputId)->info;
  }

  // For testing purposes, allow setting modified regions via "modifiedInputs"
  // attribute to full regions. This allows constructing a graph which will
  // use CopyModified via the graph builder alone. This should not normally
  // be used.
  for (const auto &inIndex : modifiedInputsViaAttrs) {
    if (!input->hasIndex(inIndex)) {
      throw error("CallOp received invalid input index ({}) for "
                  "'modifiedInputs' attribute",
                  inIndex);
    } else {
      logging::warn("[CallOp] Adding modified input ({}). This should only be "
                    "used for testing purposes.",
                    inIndex);
    }
  }
}

std::unique_ptr<Op> CallOp::clone() const {
  return std::make_unique<CallOp>(*this);
}

Graph &CallOp::getCalledGraph() const { return callee.get(); }

void CallOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  SubgraphOp::appendOutlineAttributes(os);
  os.appendAttribute("callee", callee.get().id.str());
}

std::vector<const Graph *> CallOp::getCalledGraphs() const {
  return {&getCalledGraph()};
}

void CallOp::setCalledGraph(Graph &graph) { callee = graph; }

void CallOp::connectInTensor(InIndex inIndex, TensorId tenId) {
  defaultConnectInTensor(inIndex, tenId);

  // For testing purposes, allow setting modified regions via "modifiedInputs"
  // attribute to full regions. This allows constructing a graph which will
  // use CopyModified via the graph builder alone. This should not normally
  // be used.
  auto found = std::find(
      modifiedInputsViaAttrs.begin(), modifiedInputsViaAttrs.end(), inIndex);
  if (found != modifiedInputsViaAttrs.end()) {
    auto region = view::Region::getFull(input->tensor(inIndex)->info.shape());
    addModified(inIndex, {region});
  }
}

std::vector<std::unique_ptr<Op>> CallOp::getGradOps() {

  // A SubgraphOp only has one subgraph, so index is always 0.
  SubgraphIndex subgraphIndex = 0;

  // Get required info.
  auto &bwdGraph = calledGraphGradOpHelper.getBwdGraph(subgraphIndex);

  // The implementation of `subgraphInToOpInIndex` for CallOps does not depend
  // on any members. Also, our grad op is a CallOp itself. So it's safe to pass
  // `subgraphInToOpInIndex` bound to this op.
  auto bwdGraphInToGradOpInIndex =
      std::bind(&CallOp::subgraphInToOpInIndex, this, std::placeholders::_1);
  // Get info pertaining to grad op's required inputs.
  auto gradInInfo = calledGraphGradOpHelper.getCalledGraphGradInInfo(
      subgraphIndex, bwdGraphInToGradOpInIndex);

  // The implementation of `subgraphOutToOpOutIndex` for CallOps does not depend
  // on any members. Also, our grad op is a CallOp itself. So it's safe to pass
  // `subgraphOutToOpOutIndex` bound to this op.
  auto bwdGraphOutToGradOpOutIndex =
      std::bind(&CallOp::subgraphOutToOpOutIndex, this, std::placeholders::_1);
  // Get info pertaining to grad op's outputs.
  auto gradOutInfo = calledGraphGradOpHelper.getCalledGraphGradOutToNonGradIn(
      subgraphIndex, bwdGraphOutToGradOpOutIndex);

  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(
      std::make_unique<CallGradOp>(*this, bwdGraph, gradInInfo, gradOutInfo));
  return upops;
}

CallGradOp::CallGradOp(CallOp &fwdOp,
                       Graph &bwdGraph,
                       const std::vector<GradInOutMapper> &gradInInfo_,
                       const std::map<OutIndex, InIndex> &gradOutToNonGradIn_)
    : CallOp(Onnx::CustomOperators::Call_1, bwdGraph, {}, fwdOp.settings),
      gradInInfo(gradInInfo_), outInfoMap(gradOutToNonGradIn_) {}

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
    [](const OpCreatorInfo &info, Graph &graph) -> Op * {
      ONNX_NAMESPACE::GraphProto callee =
          info.attributes.getAttribute<Attributes::Graph>("callee");

      if (callee.name().empty()) {
        throw error("CallOp subgraph must be named, so that it can be "
                    "identified for re-use.");
      }

      auto &parentGraph = info.settings.graph.get();

      std::vector<TensorId> parentScopedImplicitTensorIds;
      auto implicitTensorIds = onnxutil::getImplicitTensorIds(callee);
      for (auto implicitTensorId : implicitTensorIds) {
        auto parentScopedImplicitTensorId =
            addScope(parentGraph.getScope(), implicitTensorId);
        parentScopedImplicitTensorIds.push_back(parentScopedImplicitTensorId);
      }
      logging::op::trace("[CallOp] Callee: {}, implicit tensors: {}",
                         callee.name(),
                         implicitTensorIds);

      // Adding 'modifiedInput' to CallOps to allow creation of CallOps
      // that use CopyModified copies using the builder.
      auto modifiedInputsInt64 = info.attributes.getAttribute<Attributes::Ints>(
          "modifiedInputs", std::vector<int64_t>());
      std::vector<int> modifiedInputs;
      std::transform(
          modifiedInputsInt64.begin(),
          modifiedInputsInt64.end(),
          std::back_inserter(modifiedInputs),
          [](int64_t modifiedInput) -> int { return modifiedInput; });

      // If the callee subgraph has already been constructed, get that.
      // Otherwise, construct here.
      auto &ir = info.settings.graph.get().getIr();
      Graph *calleeGraph;
      bool hasGraph = ir.hasGraph(callee.name());

      if (hasGraph) {
        calleeGraph = &ir.getGraph(callee.name());
      } else {
        calleeGraph = &ir.createGraph(callee.name());
      }

      Op *op = graph.createOp<CallOp>(
          info.opid, *calleeGraph, modifiedInputs, info.settings);
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
        op->connectInTensor(op->input->n(), parentScopedImplicitTensorId);
      }

      if (!hasGraph) {
        // Find the input tensors in the parent graph (or its called
        // graphs) to determine the tensor type
        if (info.hasInputIds()) {
          auto inputs = info.getInputIds();
          std::map<TensorId, TensorInfo> inputInfos;
          for (auto &graph : ir.getAllGraphs()) {
            for (TensorId input : inputs) {
              if (graph->getTensors().contains(input, graph->getScope())) {
                TensorId tid =
                    graph->getTensors().find(input, graph->getScope());
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

          // Explicit inputs
          for (int i = 0; i < callee.input_size(); i++) {
            // Assume callee graph inputs are in the same order as this
            // op's node inputs
            TensorInfo calleeInputInfo = inputInfos.at(inputs.at(i));
            auto scopedId =
                addScope(calleeGraph->getScope(), callee.input(i).name());
            calleeGraph->addInput(scopedId, calleeInputInfo);
          }
        }

        // Implicit inputs
        for (auto implicitTensorId : implicitTensorIds) {
          auto parentScopedImplicitTensorId =
              addScope(parentGraph.getScope(), implicitTensorId);
          Tensor *tensor =
              parentGraph.getTensors().get(parentScopedImplicitTensorId);
          auto calleeScopedId =
              addScope(calleeGraph->getScope(), implicitTensorId);
          calleeGraph->addInput(calleeScopedId, tensor->info);
          logging::op::trace("[CallOp] Callee: {}, implicit tensor: {} -> {}",
                             callee.name(),
                             parentScopedImplicitTensorId,
                             calleeScopedId);
        }

        calleeGraph->constructFromOnnxGraph(callee);

        for (auto &output : callee.output()) {
          auto scopedId = addScope(calleeGraph->getScope(), output.name());
          calleeGraph->markAsOutput(scopedId);
        }
      }

      // If pipelining is enabled, subgraph ops must have a virtual graph id and
      // it must match the calling ops virtual graph id.
      // If in the future we allow pipeline stages to span multiple virtual
      // graphs, this will have to change to check that the subgraph op has a
      // valid virtual graph id for the pipeline stage it is called from.
      if (ir.getSessionOptions().enablePipelining) {
        VGraphId vgid = *info.settings.vgraphId;
        for (auto &id_op : calleeGraph->getOps()) {
          auto op = id_op.second.get();
          if (!op->hasVirtualGraphId()) {
            throw error(
                "Op {} in subgraph \"{}\" does not have a virtual graph id. "
                "When "
                "pipelining, subgraph ops must have a virtual graph id set.",
                op->debugName(),
                callee.name());
          } else if (op->getVirtualGraphId() != vgid) {
            throw error(
                "The virtual graph id ({}) for Op {} in subgraph \"{}\" "
                "does not match the virtual graph id ({}) of the "
                "calling op. When pipelining, subgraph ops must have a "
                "virtual graph id matching the calling op. If you are trying "
                "to call the same subgraph from different pipeline stages, you "
                "will need to create a separate subgraph for each virtual "
                "graph.",
                op->getVirtualGraphId(),
                op->debugName(),
                callee.name(),
                vgid);
          }
        }
      }

      // Connect outputs
      if (info.hasOutputIds()) {
        for (OutIndex i = 0; i < info.getOutputIds().size(); ++i) {
          op->createAndConnectOutTensor(i, info.getOutputIds().at(i));
        }
      }

      return op;
    },
    true);
} // namespace

} // namespace popart
