// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <popart/graph.hpp>
#include <popart/op/loop.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/loopx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <snap/poputil/TileMapping.hpp>

namespace popart {
namespace popx {

LoopOpx::LoopOpx(Op *op, Devicex *devicex) : SubgraphOpx(op, devicex) {
  verifyOp<LoopOp>(op, {Onnx::Operators::Loop_1, Onnx::Operators::Loop_11});
}

InputCreatorType LoopOpx::getInputCreatorType(InIndex index) const {
  if (index >= 2) {
    return InputCreatorType::CanDelegateOrUnwind;
  } else {
    return InputCreatorType::CanDelegate;
  }
}

bool LoopOpx::canUnwind(InIndex in, OutIndex out) const {
  auto &op = getOp<LoopOp>();
  if (out >= 0 && out < op.output->n() && in > 1 && in - 2 == out) {
    return true;
  }
  return false;
}

snap::Tensor LoopOpx::unwindTensorLayout(snap::Tensor tensor,
                                         InIndex in,
                                         OutIndex out) const {
  if (canUnwind(in, out)) {
    return tensor;
  } else {
    throw error("[LoopOpx] Unwinding from output {} to input {} not supported",
                out,
                in);
  }
}

view::RegMap LoopOpx::unwindRegion(InIndex in, OutIndex out) const {
  if (canUnwind(in, out)) {
    return [](const view::Region &r) { return view::Regions(1, r); };
  } else {
    throw error("[LoopOpx] Unwinding from output {} to input {} not supported",
                out,
                in);
  }
}

std::vector<snap::Tensor> LoopOpx::cloneBodyOutputs() const {
  std::vector<snap::Tensor> tensors;
  auto &op       = getOp<LoopOp>();
  auto &subgraph = op.getCalledGraph();
  for (auto outId : subgraph.getOutputIds()) {
    auto bodyOutputTensor = get(outId);
    auto clonedBodyOutputTensor =
        graph().clone(bodyOutputTensor, debugContext(outId + "_tmpClone"));
    tensors.push_back(clonedBodyOutputTensor);
  }
  return tensors;
}

void LoopOpx::copyExplicitOpInputsToBodyOutputs(
    snap::program::Sequence &prog,
    std::vector<snap::Tensor> &clonedBodyOutputs) const {
  auto &op       = getOp<LoopOp>();
  auto &subgraph = op.getCalledGraph();

  // Op input 1   ->  Body output 0
  // Op input 2   ->  Body output 1
  // ..
  // Op input M   ->  Body output M-1
  for (int i = 0; i < op.getCalledGraph().getOutputIds().size(); ++i) {
    if (hasInput(i + 1)) {

      TensorId opInId    = op.inId(i + 1);
      TensorId bodyOutId = subgraph.getOutputId(i);

      auto opInputTensor = getInTensor(i + 1);

      snap::Tensor bodyOutputTensor = clonedBodyOutputs.at(i);

      auto aliases =
          dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
              {op.getIr().getTensor(op.inId(i + 1))}, true);

      bool aliased =
          aliases.find(op.getIr().getTensor(subgraph.getOutputId(i))) !=
          aliases.end();

      if (dv_p->lowering().getAliasZeroCopy()->copyInputRequired(&op, i + 1) ||
          dv_p->lowering().getAliasZeroCopy()->copyOutputRequired(
              &op, op.subgraphOutToOpOutIndex(i))) {
        snap::program::Copy copyProg(
            opInputTensor, bodyOutputTensor, false, debugContext("inputs"));
        prog.add(copyProg);
        logging::opx::trace(
            "[LoopOpx] explicit input {} -> output {} copied (aliased: {})",
            opInId,
            bodyOutId,
            aliased);
      } else {
        logging::opx::trace("[LoopOpx] explicit input {} -> output {} skipped",
                            opInId,
                            bodyOutId);
      }
    }
  }
}

void LoopOpx::copyImplicitOpInputsToImplicitBodyInputs(
    snap::program::Sequence &prog) const {
  auto &op       = getOp<LoopOp>();
  auto &subgraph = op.getCalledGraph();

  // Op input M+1   ->  Body input M+1
  // Op input M+2   ->  Body input M+2
  // ..
  // Op input N     ->  Body input N
  for (int i = op.getNumExplicitInputs();
       i < op.getNumExplicitInputs() + op.getNumImplicitInputs();
       ++i) {

    auto aliases = dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
        {op.inTensor(i)}, true);

    TensorId opInId   = op.inId(i);
    TensorId bodyInId = subgraph.getInputId(i);

    auto opInputTensor   = getInTensor(i);
    auto bodyInputTensor = get(bodyInId);

    if (aliases.find(op.getIr().getTensor(subgraph.getInputId(i))) ==
        aliases.end()) {
      if (dv_p->lowering().getAliasZeroCopy()->copyInputRequired(&op, i)) {
        snap::program::Copy copyProg(
            opInputTensor, bodyInputTensor, false, debugContext("inputs"));
        prog.add(copyProg);
        logging::opx::trace(
            "[LoopOpx] implicit input {} -> implicit input {} copied",
            opInId,
            bodyInId);
      } else {
        logging::opx::trace(
            "[LoopOpx] implicit input {} -> implicit input {} skipped",
            opInId,
            bodyInId);
      }
    } else {
      logging::opx::trace(
          "[LoopOpx] implicit input {} -> implicit input {} aliased",
          opInId,
          bodyInId);
    }
  }
}

void LoopOpx::copyBodyOutputsToBodyOutputClones(
    snap::program::Sequence &prog,
    std::vector<snap::Tensor> &clonedBodyOutputs) const {
  auto &op       = getOp<LoopOp>();
  auto &subgraph = op.getCalledGraph();
  for (int i = 0; i < op.getCalledGraph().getOutputIds().size(); ++i) {
    TensorId outId              = subgraph.getOutputId(i);
    auto bodyOutputTensor       = get(outId);
    auto clonedBodyOutputTensor = clonedBodyOutputs.at(i);

    prog.add(
        snap::program::Copy(bodyOutputTensor,
                            clonedBodyOutputTensor,
                            false,
                            debugContext("copyBodyOutputsToBodyOutputClones")));
  }
}

void LoopOpx::copyBodyOutputsToExplicitBodyInputs(
    snap::program::Sequence &prog,
    std::vector<snap::Tensor> &clonedBodyOutputs) const {
  auto &op       = getOp<LoopOp>();
  auto &subgraph = op.getCalledGraph();

  snap::program::Sequence copiesProg(graph());

  // Skip the trip count tensor
  // Body output 0   ->  Body input 1
  // Body output 1   ->  Body input 2
  // ..
  // Body output M-1 ->  Body input M
  for (int i = 0; i < op.getCalledGraph().getOutputIds().size(); ++i) {
    TensorId inId  = subgraph.getInputId(i + 1);
    TensorId outId = subgraph.getOutputId(i);

    auto aliases = dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
        {op.getIr().getTensor(outId)}, true);

    bool aliased = (inId == outId ||
                    aliases.find(op.getIr().getTensor(inId)) != aliases.end());
    if (dv_p->lowering().getAliasZeroCopy()->copyLoopCarriedRequired(&op, i)) {
      auto bodyInputTensor = get(inId);

      // Use clones to avoid issues with implicit aliases
      auto clonedBodyOutput = clonedBodyOutputs.at(i);

      copiesProg.add(snap::program::Copy(
          clonedBodyOutput,
          bodyInputTensor,
          false,
          debugContext("copyBodyOutputsToExplicitBodyInputs")));
      logging::opx::trace(
          "[LoopOpx] output {} -> explicit input {} copied (aliased: {})",
          outId,
          inId,
          aliased);
    } else {
      logging::opx::trace(
          "[LoopOpx] output {} -> explicit input {} skipped", outId, inId);
    }
  }
  prog.add(copiesProg);
}

void LoopOpx::copyBodyOutputsToOpOutputs(
    snap::program::Sequence &prog,
    std::vector<snap::Tensor> &clonedBodyOutputs) const {
  auto &op = getOp<LoopOp>();

  // Skip the cond-out tensor
  // Body out 0   ->  skip
  // Body out 1   ->  Loop out 0
  // ..
  // Body out M-1 ->  Loop out M-2
  for (int i = 1; i < op.getCalledGraph().getOutputIds().size(); ++i) {
    TensorId bodyOutputTensorId = op.getCalledGraph().getOutputId(i);
    TensorId opOutputTensorId   = outId(i - 1);

    auto aliases = dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
        {op.getIr().getTensor(bodyOutputTensorId)}, true);
    bool aliased =
        aliases.find(op.getIr().getTensor(opOutputTensorId)) != aliases.end();

    if (dv_p->lowering().getAliasZeroCopy()->copyOutputRequired(&op, i - 1)) {
      auto bodyOutputTensor = clonedBodyOutputs.at(i);
      auto opOutputTensor   = get(opOutputTensorId);
      snap::program::Copy copyProg(
          bodyOutputTensor, opOutputTensor, false, debugContext("outputs"));
      prog.add(copyProg);
      logging::opx::trace(
          "[LoopOpx] output {} -> output {} copied (aliased: {})",
          bodyOutputTensorId,
          opOutputTensorId,
          aliased);
    } else {
      logging::opx::trace("[LoopOpx] output {} -> output {} skipped",
                          bodyOutputTensorId,
                          opOutputTensorId);
    }
  }
}

void LoopOpx::copyModifiedBodyInputsToOpInputs(
    snap::program::Sequence &prog) const {
  auto &op                = getOp<LoopOp>();
  const auto &graph       = op.getGraph();
  const auto &calledGraph = op.getCalledGraph();

  for (auto input : op.input->tensorIdMap()) {
    InIndex opInIndex   = input.first;
    TensorId opTensorId = input.second;
    InIndex sgInIndex   = op.opInToSubgraphInIndex(opInIndex);
    TensorId sgTensorId = calledGraph.getInputId(sgInIndex);

    auto modifiedRegions = op.modifies(opInIndex);

    if (std::any_of(modifiedRegions.begin(),
                    modifiedRegions.end(),
                    [](const view::Region &r) { return !r.isEmpty(); })) {
      auto opTensor = graph.getTensors().get(opTensorId);
      auto sgTensor = calledGraph.getTensors().get(sgTensorId);

      auto opPopTensor = get(opTensorId);
      auto sgPopTensor = get(sgTensorId);

      auto aliases =
          dv_p->lowering().getAliasZeroCopy()->getActiveAliasedTensors(
              {opTensor}, true);

      bool copy_modified_required = true;

      copy_modified_required &= aliases.find(sgTensor) == aliases.end();

      copy_modified_required &=
          dv_p->lowering().getAliasZeroCopy()->copyModifiedRequired(&op,
                                                                    opInIndex);

      if (copy_modified_required) {
        logging::opx::trace(
            "[CallOpx] Copying modified input {}->{}", sgTensorId, opTensorId);
        snap::program::Copy copy_prog(
            sgPopTensor, opPopTensor, false, debugContext());
        prog.add(copy_prog);
      } else {
        logging::opx::trace("[CallOpx] Skipping copy modified input {}->{}",
                            sgTensorId,
                            opTensorId);
      }
    }
  }
}

void LoopOpx::grow(snap::program::Sequence &prog) const {
  // Builds the logic for loops (pseudocode):
  //
  // copyExplicitOpInputsToBodyOutputs(); // will set condOut
  // copyImplicitOpInputsToBodyInputs();
  // exit = false;
  // for (i = 0; i < maxTripCount; ++i) {
  //   // loopProg
  //   exit = exit || !condOut || i >= maxTripCount);
  //   if (exit) {
  //     // loopExitProg
  //   } else {
  //     // loopContinueProg
  //     copyBodyOutputsToExplicitBodyInputs();
  //     body(); // can update condOut
  //   }
  // }
  // copyBodyOutputsToOpOutputs();
  // copyModifiedBodyInputsToOpInputs();
  //

  auto &op = getOp<LoopOp>();

  if (op.getNumImplicitScanOutputs()) {
    throw error("[LoopOpx] numImplicitScanOutputs > 0 not supported. "
                "Implicit scan outputs should have been removed by "
                "LoopScanOutPattern.");
  }

  auto tconst = getConst(poplar::BOOL, {}, true, "tconst");

  auto fconst = getConst(poplar::BOOL, {}, false, "fconst");

  // Clone body outputs
  auto bodyOutputClones = cloneBodyOutputs();

  auto condOutTensor =
      bodyOutputClones.at(LoopOp::getLoopGraphTerminationConditionOutIndex());

  // 0: Set condOut to true if the cond is not shipped as op input
  if (!hasInput(LoopOp::getTerminationConditionInIndex())) {
    prog.add(snap::program::Copy(
        tconst, condOutTensor, {}, debugContext("cond_true")));
  }

  // 1: Copy explicit inputs to body outputs
  copyExplicitOpInputsToBodyOutputs(prog, bodyOutputClones);

  // 2: Copy implicit inputs to body inputs
  copyImplicitOpInputsToImplicitBodyInputs(prog);

  // 3: Create a poplar only iterator variable i, set it to 0
  snap::Tensor iteratorTensor;
  iteratorTensor =
      graph().addVariable(poplar::INT, {}, debugContext("iterator"));
  snap::poputil::mapTensorLinearly(graph(), iteratorTensor);
  popops::zero(graph().getPoplarGraph(),
               iteratorTensor.getPoplarTensor(),
               prog.getPoplarSequence(),
               debugContext("iterator_0"));

  // 4: Create a poplar only boolean variable exit, set it to false
  auto exitTensor = graph().addVariable(poplar::BOOL, {}, debugContext("exit"));
  snap::poputil::mapTensorLinearly(graph(), exitTensor);
  prog.add(
      snap::program::Copy(fconst, exitTensor, {}, debugContext("exit_false")));

  // 5: Get the max trip count value
  auto maxTripCountValue = op.getTripCountValue();

  // 6: Create the three loop body programs
  snap::program::Sequence loopProg(debugContext("loop"), graph());
  snap::program::Sequence loopExitProg(debugContext("exit"), graph());
  snap::program::Sequence loopContinueProg(debugContext("continue"), graph());

  // 7: Update the exit condition
  if (hasInput(LoopOp::getMaximumTripCountInIndex())) {
    poplar::Tensor maxTripCountTensor =
        getInTensor(LoopOp::getMaximumTripCountInIndex()).getPoplarTensor();
    popops::mapInPlace(
        graph().getPoplarGraph(),
        popops::expr::Or(popops::expr::Or(popops::expr::_1,
                                          popops::expr::Not(popops::expr::_2)),
                         popops::expr::Gte(popops::expr::_3, popops::expr::_4)),
        {exitTensor.getPoplarTensor(),
         condOutTensor.getPoplarTensor(),
         iteratorTensor.getPoplarTensor(),
         maxTripCountTensor},
        loopProg.getPoplarSequence(),
        debugContext("exit_update"));
  } else {
    popops::mapInPlace(
        graph().getPoplarGraph(),
        popops::expr::Or(popops::expr::_1, popops::expr::Not(popops::expr::_2)),
        {exitTensor.getPoplarTensor(), condOutTensor.getPoplarTensor()},
        loopProg.getPoplarSequence(),
        debugContext("exit_update"));
  }

  // 8: Copy body outputs to body inputs
  copyBodyOutputsToExplicitBodyInputs(loopContinueProg, bodyOutputClones);

  // 9: Copy iterator to body input
  auto bodyInputTensor = get(
      op.getCalledGraph().getInputId(LoopOp::getLoopGraphIterationInIndex()));
  snap::program::Copy copyProg(
      iteratorTensor, bodyInputTensor, false, debugContext());
  loopContinueProg.add(copyProg);

  // 10: Add the loop body itself
  auto &called_graph = op.getCalledGraph();
  auto &graph_progs  = dv_p->lowering().getFragmentFunctions(called_graph);

  for (size_t part = 0; part < graph_progs.size(); ++part) {
    auto dbgStr = logging::format("{}/{}", called_graph.id.str(), part);
    loopContinueProg.add(snap::program::Call(
        graph(), graph_progs.at(part), debugContext(dbgStr)));
    copyBodyOutputsToBodyOutputClones(loopContinueProg, bodyOutputClones);
  }

  // 11: Increment the loop iterator
  popops::mapInPlace(
      graph().getPoplarGraph(),
      popops::expr::Add(popops::expr::_1, popops::expr::Const(1)),
      {iteratorTensor.getPoplarTensor()},
      loopContinueProg.getPoplarSequence(),
      debugContext("iterator_update"));

  // Test if the loop continue condition will not change, and if the loop
  // will always run until the max iteration count
  if (op.getCalledGraph().getInputId(1) == op.getCalledGraph().getOutputId(0) &&
      !op.hasInput(LoopOp::getMaximumTripCountInIndex()) &&
      !op.hasInput(LoopOp::getTerminationConditionInIndex())) {
    // 12: Add loop body program directly
    loopProg.add(loopContinueProg);
  } else {
    // 12: Add conditional around the loop body program
    loopProg.add(snap::program::If(
        exitTensor, loopExitProg, loopContinueProg, debugContext("condition")));
  }

  // 13: Repeat the loop conditional program
  logging::opx::debug(
      "[LoopOpx] Max trip count: {} ({})", maxTripCountValue, op.debugName());
  prog.add(
      snap::program::Repeat(maxTripCountValue, loopProg, debugContext("loop")));

  // 14: Copy body outputs to op outputs
  copyBodyOutputsToOpOutputs(prog, bodyOutputClones);

  // 15: Copy modified inputs
  copyModifiedBodyInputsToOpInputs(prog);
}

namespace {
OpxCreator<LoopOpx> loopOpxCreator({Onnx::Operators::Loop_1,
                                    Onnx::Operators::Loop_11});
} // namespace
} // namespace popx
} // namespace popart
