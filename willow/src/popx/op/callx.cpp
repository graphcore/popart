// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <popart/aliaszerocopy.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/call.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/callx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

CallOpx::CallOpx(Op *op, Devicex *devicex) : SubgraphOpx(op, devicex) {
  verifyOp<CallOp>(op, Onnx::CustomOperators::Call_1);
}

InputCreatorType CallOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanDelegate;
}

void CallOpx::copyModified(poplar::program::Sequence &prog) const {
  auto &callop = getOp<CallOp>();

  for (int i = 0; i < callop.input->n(); i++) {
    auto modifiedRegions    = callop.modifies(i);
    const auto &calledGraph = callop.getCalledGraph();
    TensorId graph_input_id = calledGraph.getInputId(i);
    if (std::any_of(modifiedRegions.begin(),
                    modifiedRegions.end(),
                    [](const view::Region &r) { return !r.isEmpty(); })) {
      TensorId call_input_id = callop.inId(i);
      auto call_input        = get(call_input_id);
      auto graph_input       = get(graph_input_id);
      auto aliases =
          dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
              {callop.input->tensor(i)}, true);

      bool copy_modified_required = true;

      copy_modified_required &=
          aliases.find(callop.getIr().getTensor(graph_input_id)) ==
          aliases.end();

      copy_modified_required &=
          dv_p->lowering().getAliasZeroCopy()->copyModifiedRequired(&callop, i);

      if (copy_modified_required) {
        logging::opx::trace("[CallOpx] Copying modified input {}->{}",
                            graph_input_id,
                            callop.inId(i));
        poplar::program::Copy copy_prog(
            graph_input, call_input, false, debugContext());
        prog.add(copy_prog);
      } else {
        logging::opx::trace("[CallOpx] Skipping copy modified input {}->{}",
                            graph_input_id,
                            callop.inId(i));
      }
    }
  }
}

void CallOpx::copyInputs(poplar::program::Sequence &prog) const {
  auto &callop = getOp<CallOp>();

  for (int i = 0; i < callop.input->n(); i++) {
    TensorId call_input_id  = callop.inId(i);
    auto call_input         = get(call_input_id);
    TensorId graph_input_id = callop.getCalledGraph().getInputId(i);
    auto graph_input        = get(graph_input_id);

    auto aliases = dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
        {callop.input->tensor(i)}, true);

    view::AccessType accessType = view::AccessType::None;
    for (auto &r : callop.modifies(i)) {
      accessType = view::combine({accessType, r.getAccessType()});
    }

    // Only copy inputs if
    // a.) It is not aliased (no call by reference; call by value instead)
    // b.) The access type is not write-only (at least one consumer of the
    //     unmodified input tensor in the subgraph will read the contents
    //     of the tensor)
    if (aliases.find(callop.getIr().getTensor(graph_input_id)) ==
        aliases.end()) {
      if (accessType != view::AccessType::Write &&
          dv_p->lowering().getAliasZeroCopy()->copyInputRequired(&callop, i)) {
        logging::opx::trace(
            "[CallOpx] Copying input {}->{}", call_input_id, graph_input_id);
        poplar::program::Copy copy_prog(
            call_input, graph_input, false, debugContext());
        prog.add(copy_prog);
      } else {
        logging::opx::trace("[CallOpx] Skipping copy input {}->{} "
                            "(tensor not read in subgraph)",
                            call_input_id,
                            graph_input_id);
      }
    } else {
      logging::opx::trace(
          "[CallOpx] Aliasing input {}->{}", call_input_id, graph_input_id);
    }
    if (accessType == view::AccessType::Write) {
      logging::opx::trace("[CallOpx] Write undef tensor {}", graph_input_id);
      prog.add(poplar::program::WriteUndef(graph_input));
    }
  }
}

void CallOpx::copyOutputs(poplar::program::Sequence &prog) const {
  auto &callop = getOp<CallOp>();
  for (int i = 0; i < callop.output->n(); i++) {
    TensorId call_output_id = callop.outId(i);
    auto call_output        = getOutTensor(i);
    auto graph_output_id    = callop.getCalledGraph().getOutputId(i);
    auto graph_output       = get(graph_output_id);

    auto aliases = dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
        {callop.getIr().getTensor(graph_output_id)}, true);

    // Post IR aliased between subgraph output and CallOp output
    bool aliased = (aliases.find(callop.getIr().getTensor(call_output_id)) !=
                    aliases.end());

    for (int j = 0; j < callop.input->n(); j++) {
      auto input = get(callop.inId(j));
      // Fully aliased from CallOp input to CallOp output & shape did not change
      auto aliasRegions = callop.aliases(j, i);

      // Aliased from op input to op output
      bool alias = aliasRegions.size() == 1 &&
                   aliasRegions.front().nelms() == call_output.numElements() &&
                   call_output.shape() == input.shape();
      aliased |= alias;
    }

    // Skip copy if aliased tensor -> is handled by copyModified, or is aliased
    // from graph output to op output
    if (!aliased) {
      if (dv_p->lowering().getAliasZeroCopy()->copyOutputRequired(&callop, i)) {
        logging::opx::trace("[CallOpx] Copying output {}->{}",
                            graph_output_id,
                            callop.outId(i));
        poplar::program::Copy copy_prog(
            graph_output, call_output, false, debugContext());
        prog.add(copy_prog);
      } else {
        logging::opx::trace("[CallOpx] Skipping output {}->{}",
                            graph_output_id,
                            callop.outId(i));
      }
    } else {
      logging::opx::trace(
          "[CallOpx] Aliasing output {}->{}", graph_output_id, callop.outId(i));
    }
  }
}

void CallOpx::doCall(poplar::program::Sequence &prog) const {
  auto &callop       = getOp<CallOp>();
  auto &called_graph = callop.getCalledGraph();
  auto &graph_prog   = dv_p->lowering().getFragmentFunction(called_graph);
  prog.add(poplar::program::Call(graph_prog, debugContext()));
}

void CallOpx::grow(poplar::program::Sequence &prog) const {
  copyInputs(prog);
  doCall(prog);
  copyOutputs(prog);
  copyModified(prog);
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
