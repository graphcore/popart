#include <popart/graph.hpp>
#include <popart/op/call.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/callx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

CallOpx::CallOpx(Op *op, Devicex *devicex) : SubgraphOpx(op, devicex) {
  verifyOp<CallOp>(op, Onnx::CustomOperators::Call_1);
}

std::pair<std::vector<ICreatorCandidatePtr>, std::vector<UnwindEndpointPtr>>
CallOpx::getEndpoints(InIndex index, std::vector<OpxInAndOutIndex> path) const {
  auto &callop      = getOp<CallOp>();
  auto &callgraph   = callop.getCalledGraph();
  auto in_tensor_id = callgraph.getInputId(index);
  auto inTensor     = callgraph.getTensors().get(in_tensor_id);

  // Internal endpoints
  auto endpoints = dv_p->getCreatorEndpoints(inTensor, path);

  return endpoints;
}

InputCreatorType CallOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANDELEGATE;
}

void CallOpx::copyModified(poplar::program::Sequence &prog) const {
  auto &callop = getOp<CallOp>();

  for (int i = 0; i < callop.input->n(); i++) {
    if (callop.isInputModified(i)) {
      auto call_input         = get(callop.inId(i));
      auto graph_input_id     = callop.getCalledGraph().getInputId(i);
      auto graph_input        = get(graph_input_id);
      const auto &calledGraph = callop.getCalledGraph();
      if (!calledGraph.isMarkedAsZeroCopy(graph_input_id) &&
          !calledGraph.isInputConsumedInplaceForOptimization(graph_input_id)) {
        logging::trace("[CallOpx] Copying modified input {}->{}",
                       graph_input_id,
                       callop.inId(i));
        poplar::program::Copy copy_prog(graph_input, call_input);
        prog.add(copy_prog);
      }
    }
  }
}

void CallOpx::copyInputs(poplar::program::Sequence &prog) const {
  auto &callop = getOp<CallOp>();

  for (int i = 0; i < callop.input->n(); i++) {
    auto call_input     = get(callop.inId(i));
    auto graph_input_id = callop.getCalledGraph().getInputId(i);
    auto graph_input    = get(graph_input_id);
    if (!callop.getCalledGraph().isMarkedAsZeroCopy(graph_input_id)) {
      logging::trace(
          "[CallOpx] Copying input {}->{}", callop.inId(i), graph_input_id);
      poplar::program::Copy copy_prog(call_input, graph_input);
      prog.add(copy_prog);
    }
  }
}

void CallOpx::copyOutputs(poplar::program::Sequence &prog) const {
  auto &callop = getOp<CallOp>();
  for (int i = 0; i < callop.output->n(); i++) {
    auto call_output     = getOutTensor(i);
    auto graph_output_id = callop.getCalledGraph().getOutputId(i);
    auto graph_output    = get(graph_output_id);

    bool aliased = false;
    for (int j = 0; j < callop.input->n(); j++) {
      auto input = get(callop.inId(j));
      // Fully aliased & shape did not change
      auto aliasRegions = callop.aliases(j, i);
      bool alias        = aliasRegions.size() == 1 &&
                   aliasRegions.front().nelms() == call_output.numElements() &&
                   call_output.shape() == input.shape();
      aliased |= alias;
    }

    // Skip copy if aliased tensor
    if (!aliased) {
      logging::trace(
          "[CallOpx] Copying output {}->{}", graph_output_id, callop.outId(i));
      poplar::program::Copy copy_prog(graph_output, call_output);
      prog.add(copy_prog);
    } else {
      logging::trace("[CallOpx] Skipping aliased output {}->{}",
                     graph_output_id,
                     callop.outId(i));
    }
  }
}

void CallOpx::doCall(poplar::program::Sequence &prog) const {
  auto &callop       = getOp<CallOp>();
  auto &called_graph = callop.getCalledGraph();
  auto &graph_prog   = dv_p->getFragmentFunction(called_graph);
  prog.add(poplar::program::Call(graph_prog));
}

void CallOpx::grow(poplar::program::Sequence &prog) const {
  copyInputs(prog);
  doCall(prog);
  copyOutputs(prog);
  copyModified(prog);
}

std::vector<std::tuple<TensorId, TensorId, bool>>
CallOpx::getOutputsToPrepare() const {
  auto &callop = getOp<CallOp>();
  std::vector<std::tuple<TensorId, TensorId, bool>> outputs;
  int i = 0;
  for (auto subgraph_out_id : callop.getCalledGraph().getOutputIds()) {
    bool aliased = false;
    for (int j = 0; j < callop.input->n(); j++) {
      // Fully aliased & shape did not change
      auto aliasRegions = callop.aliases(j, i);
      bool alias        = aliasRegions.size() == 1 &&
                   aliasRegions.front().nelms() ==
                       callop.output->tensor(i)->info.nelms() &&
                   callop.output->tensor(i)->info.shape() ==
                       callop.input->tensor(j)->info.shape();
      aliased |= alias;
      if (alias)
        subgraph_out_id = callop.input->id(j);
    }

    TensorId call_out_id = callop.output->tensor(i)->id;

    logging::trace(
        "To prepare graph output {}, aliased: {}", subgraph_out_id, aliased);
    outputs.emplace_back(subgraph_out_id, call_out_id, aliased);
    ++i;
  }
  return outputs;
}

CallGradOpx::CallGradOpx(Op *op, Devicex *devicex) : CallOpx(op, devicex) {
  verifyOp<CallGradOp>(op, Onnx::CustomGradOperators::CallGrad);
}

namespace {
OpxCreator<CallOpx> callOpxCreator(Onnx::CustomOperators::Call_1);
OpxCreator<CallGradOpx> callGradOpxCreator(Onnx::CustomGradOperators::CallGrad);
} // namespace

} // namespace popx
} // namespace popart
