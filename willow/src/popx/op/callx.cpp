#include <popart/graph.hpp>
#include <popart/op/call.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/callx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

CallOpx::CallOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<CallOp>(op, Onnx::CustomOperators::Call);
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

std::vector<std::pair<poplar::Tensor, bool>> CallOpx::prepareOutputs() const {
  std::vector<std::pair<poplar::Tensor, bool>> outputs;
  auto &callop = getOp<CallOp>();

  int i = 0;
  for (auto out_id : callop.getCalledGraph().getOutputIds()) {
    auto output = get(out_id);

    bool aliased = false;
    for (int j = 0; j < callop.input->n(); j++) {
      auto input = get(callop.inId(j));
      // Fully aliased & shape did not change
      auto aliasRegions = callop.aliases(j, i);
      bool alias        = aliasRegions.size() == 1 &&
                   aliasRegions.front().nelms() == output.numElements() &&
                   output.shape() == input.shape();
      aliased |= alias;
      if (alias) {
        // Set aliased input as output
        output = input;
      }
    }

    logging::trace("Preparing graph output {}, aliased: {}", out_id, aliased);
    outputs.push_back({aliased ? output : graph().clone(output), aliased});
    ++i;
  }

  return outputs;
}

void CallOpx::copyModified(poplar::program::Sequence &prog) const {
  auto &callop = getOp<CallOp>();

  for (int i = 0; i < callop.input->n(); i++) {
    if (callop.isInputModified(i)) {
      auto call_input     = get(callop.inId(i));
      auto graph_input_id = callop.getCalledGraph().getInputId(i);
      auto graph_input    = get(graph_input_id);
      if (!callop.getCalledGraph().isMarkedAsZeroCopy(graph_input_id)) {
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

void CallOpx::copyOutputs(
    poplar::program::Sequence &prog,
    const std::vector<std::pair<poplar::Tensor, bool>> &outputs) const {
  auto &callop = getOp<CallOp>();
  for (int i = 0; i < outputs.size(); i++) {
    auto &call_output    = outputs.at(i).first;
    bool aliased         = outputs.at(i).second;
    auto graph_output_id = callop.getCalledGraph().getOutputId(i);
    auto graph_output    = get(graph_output_id);

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
  auto outputs = prepareOutputs();
  copyOutputs(prog, outputs);
  copyModified(prog);
  for (int i = 0; i < outputs.size(); i++) {
    setOutTensor(i, outputs.at(i).first);
  }
}

namespace {
OpxCreator<CallOpx> callOpxCreator(Onnx::CustomOperators::Call);
}

} // namespace popx
} // namespace popart
