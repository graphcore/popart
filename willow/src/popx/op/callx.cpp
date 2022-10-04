// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <ostream>
#include <set>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <tuple>
#include <vector>
#include <popart/aliaszerocopy.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/call.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/callx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/subgraphpartitioner.hpp>
#include <popart/tensorindex.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/pointercomparators.hpp"
#include "popart/popx/op/subgraphx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class Tensor;

namespace popx {

CallOpx::CallOpx(Op *op, Devicex *devicex) : SubgraphOpx(op, devicex) {
  verifyOp<CallOp>(op, Onnx::CustomOperators::Call_1);
}

InputCreatorType CallOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanDelegate;
}

void CallOpx::copyModified(snap::program::Sequence &prog,
                           InIndex inputIndex) const {
  auto &callop = getOp<CallOp>();
  auto &i      = inputIndex;

  auto modifiedRegions    = callop.modifies(i);
  const auto &calledGraph = callop.getCalledGraph();
  TensorId graph_input_id = calledGraph.getInputId(i);

  if (std::any_of(modifiedRegions.begin(),
                  modifiedRegions.end(),
                  [](const view::Region &r) { return !r.isEmpty(); })) {
    TensorId call_input_id = callop.inId(i);
    auto call_input        = get(call_input_id);
    auto graph_input       = get(graph_input_id);

    auto aliases = dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
        {callop.input->tensor(i)}, true);

    bool copy_modified_required = true;

    copy_modified_required &=
        aliases.find(callop.getIr().getTensor(graph_input_id)) == aliases.end();

    copy_modified_required &=
        dv_p->lowering().getAliasZeroCopy()->copyModifiedRequired(&callop, i);

    if (copy_modified_required) {
      logging::opx::trace("[CallOpx] Copying modified input {}->{}",
                          graph_input_id,
                          callop.inId(i));
      snap::program::Copy copy_prog(
          graph_input, call_input, false, debugContext());
      prog.getPoplarSequence().add(copy_prog);
    } else {
      logging::opx::trace("[CallOpx] Skipping copy modified input {}->{}",
                          graph_input_id,
                          callop.inId(i));
    }
  }
}

void CallOpx::copyInput(snap::program::Sequence &prog,
                        InIndex inputIndex) const {
  auto &callop = getOp<CallOp>();
  auto &i      = inputIndex;

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
  if (aliases.find(callop.getIr().getTensor(graph_input_id)) == aliases.end()) {
    if (accessType != view::AccessType::Write &&
        dv_p->lowering().getAliasZeroCopy()->copyInputRequired(&callop, i)) {
      logging::opx::trace(
          "[CallOpx] Copying input {}->{}", call_input_id, graph_input_id);
      snap::program::Copy copy_prog(
          call_input, graph_input, false, debugContext());
      prog.getPoplarSequence().add(copy_prog);
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
    prog.getPoplarSequence().add(
        snap::program::WriteUndef(graph_input, debugContext()));
  }
}

void CallOpx::copyOutput(snap::program::Sequence &prog,
                         OutIndex outputIndex) const {
  auto &callop = getOp<CallOp>();
  auto &i      = outputIndex;

  TensorId call_output_id = callop.outId(i);
  auto call_output        = getOutTensor(i);
  auto graph_output_id    = callop.getCalledGraph().getOutputId(i);
  auto graph_output       = get(graph_output_id);

  auto aliases = dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
      {callop.getIr().getTensor(graph_output_id)}, true);

  // Post IR aliased between subgraph output and CallOp output
  bool aliased =
      (aliases.find(callop.getIr().getTensor(call_output_id)) != aliases.end());

  for (int j = 0; j < callop.input->n(); j++) {
    auto input = get(callop.inId(j));
    // Fully aliased from CallOp input to CallOp output & shape did not
    // change
    auto aliasRegions = callop.aliases(j, i);

    // Aliased from op input to op output
    bool alias = aliasRegions.size() == 1 &&
                 aliasRegions.front().nelms() == call_output.numElements() &&
                 call_output.shape() == input.shape();
    aliased |= alias;
  }

  // Skip copy if aliased tensor -> is handled by copyModified, or is
  // aliased from graph output to op output
  if (!aliased) {
    if (dv_p->lowering().getAliasZeroCopy()->copyOutputRequired(&callop, i)) {
      logging::opx::trace(
          "[CallOpx] Copying output {}->{}", graph_output_id, callop.outId(i));
      snap::program::Copy copy_prog(
          graph_output, call_output, false, debugContext());
      prog.getPoplarSequence().add(copy_prog);
    } else {
      logging::opx::trace(
          "[CallOpx] Skipping output {}->{}", graph_output_id, callop.outId(i));
    }
  } else {
    logging::opx::trace(
        "[CallOpx] Aliasing output {}->{}", graph_output_id, callop.outId(i));
  }
}

void CallOpx::doCall(snap::program::Sequence &prog,
                     SubgraphPartIndex subgraphPart) const {
  auto &callop       = getOp<CallOp>();
  auto &called_graph = callop.getCalledGraph();
  auto &graph_prog =
      dv_p->lowering().getFragmentFunction(called_graph, subgraphPart);

  logging::opx::trace("[CallOpx] Calling {}, part {}",
                      called_graph.getGraphString(),
                      subgraphPart);
  auto dbgStr = logging::format("{}/{}", called_graph.id.str(), subgraphPart);
  prog.getPoplarSequence().add(
      snap::program::Call(graph(), graph_prog, debugContext(dbgStr)));
}

void CallOpx::grow(std::vector<snap::program::Sequence> &sequences) const {

  auto partitioner    = dv_p->lowering().getSubgraphPartitioner();
  auto &callop        = getOp<CallOp>();
  auto callOpSchedule = partitioner->getCallOpSchedule(&callop);

  // Used to keep track of which fragment of the called subgraph to call.
  int calledSubgraphPart = 0;

  if (!callOpSchedule.empty()) {

    // Used to determine a relative offset from first fragment this op affects.
    int offsetSubgraphPart = std::get<1>(callOpSchedule.front());

    for (const auto &tuple : callOpSchedule) {
      const auto &callOpPart   = std::get<0>(tuple);
      int relativeSubgraphPart = std::get<1>(tuple) - offsetSubgraphPart;

      // Make sure we have enough sequences to lower this in.
      while (sequences.size() <= relativeSubgraphPart) {
        int subgraphPart = (sequences.size() + offsetSubgraphPart);
        std::stringstream ss;
        ss << callop.getGraph().id.str() << "/" << subgraphPart;
        sequences.push_back({debugContext(ss.str()), dv_p->lowering().graph()});
      }

      using CallOpPartType = liveness::SubgraphPartitioner::CallOpPartType;

      // Lower different things based on liveness analysis.
      switch (callOpPart.type) {
      case CallOpPartType::CopyInput: {
        copyInput(sequences[relativeSubgraphPart], callOpPart.inIndex);
        break;
      }
      case CallOpPartType::CopyOutput: {
        copyOutput(sequences[relativeSubgraphPart], callOpPart.outIndex);
        break;
      }
      case CallOpPartType::CopyModified: {
        copyModified(sequences[relativeSubgraphPart], callOpPart.inIndex);
        break;
      }
      case CallOpPartType::CallSubgraphPart: {
        assert(callOpPart.subgraphPartIndex == calledSubgraphPart);
        doCall(sequences[relativeSubgraphPart], calledSubgraphPart++);
        break;
      }
      case CallOpPartType::Undefined:
      default:
        throw error("[CallOpx] Unexpected value of CallOpPartType");
      }
    }
  }

  // Do a sanity check -- did the CallOp call every subgraph part of the
  // called subgraph? If not, something is wrong.
  auto numCalls = partitioner->getNumSubgraphParts(callop.getCalledGraph());
  if (calledSubgraphPart != numCalls) {
    throw internal_error("[CallOpx] While lowering {} in {} {} subgraph "
                         "parts were called (expected {})",
                         callop.debugName(),
                         callop.getGraph().getGraphString(),
                         calledSubgraphPart,
                         numCalls);
  }
}

void CallOpx::grow(snap::program::Sequence &prog) const {
  throw error("growing CallOpx requires a vector of sequences {}", op_p->opid);
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
