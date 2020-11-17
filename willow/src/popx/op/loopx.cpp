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
  if (out >= 0 && out < op.output->n() && in >= 1 && in - 1 == out) {
    return true;
  }
  return false;
}

poplar::Tensor LoopOpx::unwindTensorLayout(poplar::Tensor tensor,
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

void LoopOpx::copyExplicitOpInputsToBodyOutputs(
    poplar::program::Sequence &prog) const {
  auto &op       = getOp<LoopOp>();
  auto &subgraph = op.getCalledGraph();

  // Op input 1   ->  Body output 0
  // Op input 2   ->  Body output 1
  // ..
  // Op input M   ->  Body output M-1
  for (int i = 0; i < op.getCalledGraph().getOutputIds().size(); ++i) {
    if (hasInput(i + 1)) {
      auto opInputTensor    = getInTensor(i + 1);
      auto bodyOutputTensor = get(subgraph.getOutputId(i));
      poplar::program::Copy copyProg(opInputTensor, bodyOutputTensor);
      prog.add(copyProg);
    }
  }
}

void LoopOpx::copyImplicitOpInputsToImplicitBodyInputs(
    poplar::program::Sequence &prog) const {
  auto &op       = getOp<LoopOp>();
  auto &subgraph = op.getCalledGraph();

  // Op input M+1   ->  Body input M+1
  // Op input M+2   ->  Body input M+2
  // ..
  // Op input N     ->  Body input N
  for (int i = op.numExplicitInputs();
       i < op.numExplicitInputs() + op.numImplicitInputs();
       ++i) {
    auto opInputTensor   = getInTensor(i);
    auto bodyInputTensor = get(subgraph.getInputId(i));
    poplar::program::Copy copyProg(opInputTensor, bodyInputTensor);
    prog.add(copyProg);
  }
}

void LoopOpx::copyBodyOutputsToExplicitBodyInputs(
    poplar::program::Sequence &prog) const {
  auto &op       = getOp<LoopOp>();
  auto &subgraph = op.getCalledGraph();

  // Skip the trip count tensor
  // Body output 0   ->  Body input 1
  // Body output 1   ->  Body input 2
  // ..
  // Body output M-1 ->  Body input M
  for (int i = 0; i < op.getCalledGraph().getOutputIds().size(); ++i) {
    TensorId inId  = subgraph.getInputId(i + 1);
    TensorId outId = subgraph.getOutputId(i);

    if (inId != outId) {
      // Only copy if the input is not directly wired through to the output
      auto bodyInputTensor  = get(inId);
      auto bodyOutputTensor = get(outId);
      poplar::program::Copy copyProg(bodyOutputTensor, bodyInputTensor);
      prog.add(copyProg);
    }
  }
}

void LoopOpx::copyBodyOutputsToOpOutputs(
    poplar::program::Sequence &prog) const {
  auto &op = getOp<LoopOp>();

  // Skip the cond-out tensor
  // Body out 0   ->  skip
  // Body out 1   ->  Loop out 0
  // ..
  // Body out M-1 ->  Loop out M-2
  for (int i = 1; i < op.getCalledGraph().getOutputIds().size(); ++i) {
    auto bodyOutputTensor = get(op.getCalledGraph().getOutputId(i));
    auto opOutputTensor   = get(outId(i - 1));
    poplar::program::Copy copyProg(bodyOutputTensor, opOutputTensor);
    prog.add(copyProg);
  }
}

void LoopOpx::grow(poplar::program::Sequence &prog) const {
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
  //     copyBodyOutputsToBodyInputs();
  //     body(); // can update condOut
  //   }
  // }
  // copyBodyOutputsToOpOutputs();
  //

  auto &op = getOp<LoopOp>();

  auto condOutTensor = get(op.getCalledGraph().getOutputId(0));

  // 0: Set condOut to true if the cond is not shipped as op input
  if (!hasInput(LoopOp::getTerminationConditionInIndex())) {
    popops::mapInPlace(
        graph(),
        popops::expr::And(popops::expr::_1, popops::expr::Const(true)),
        {condOutTensor},
        prog,
        debugPrefix("cond_true"));
  }

  // 1: Copy explicit inputs to body outputs
  copyExplicitOpInputsToBodyOutputs(prog);

  // 2: Copy implicit inputs to body inputs
  copyImplicitOpInputsToImplicitBodyInputs(prog);

  // 3: Create a poplar only iterator variable i, set it to 0
  poplar::Tensor iteratorTensor;
  if (hasInput(LoopOp::getMaximumTripCountInIndex())) {
    iteratorTensor =
        graph().addVariable(poplar::INT, {}, debugPrefix("iterator"));
    poputil::mapTensorLinearly(graph(), iteratorTensor);
    popops::zero(graph(), iteratorTensor, prog, debugPrefix("iterator_0"));
  }

  // 4: Create a poplar only boolean variable exit, set it to false
  auto exitTensor = graph().addVariable(poplar::BOOL, {}, debugPrefix("exit"));
  poputil::mapTensorLinearly(graph(), exitTensor);
  popops::mapInPlace(
      graph(),
      popops::expr::And(popops::expr::_1, popops::expr::Const(false)),
      {exitTensor},
      prog,
      debugPrefix("exit_false"));

  // 5: Get the max trip count value
  auto maxTripCountValue = op.getTripCountValue();

  // 6: Create the three loop body programs
  poplar::program::Sequence loopProg, loopExitProg, loopContinueProg;

  // 7: Update the exit condition
  if (hasInput(LoopOp::getMaximumTripCountInIndex())) {
    poplar::Tensor maxTripCountTensor =
        getInTensor(LoopOp::getMaximumTripCountInIndex());
    popops::mapInPlace(
        graph(),
        popops::expr::Or(popops::expr::Or(popops::expr::_1,
                                          popops::expr::Not(popops::expr::_2)),
                         popops::expr::Gte(popops::expr::_3, popops::expr::_4)),
        {exitTensor, condOutTensor, iteratorTensor, maxTripCountTensor},
        loopProg,
        debugPrefix("exit_update"));
  } else {
    popops::mapInPlace(
        graph(),
        popops::expr::Or(popops::expr::Or(popops::expr::_1,
                                          popops::expr::Not(popops::expr::_2))),
        {exitTensor, condOutTensor},
        loopProg,
        debugPrefix("exit_update"));
  }

  // 8: Copy body outputs to body inputs
  copyBodyOutputsToExplicitBodyInputs(loopContinueProg);

  // 9: Add the loop body itself
  auto &loopBody = dv_p->lowering().progs.scopeFragment(op.getCalledGraph());
  loopContinueProg.add(loopBody);

  // 10: Increment the loop iterator
  if (hasInput(LoopOp::getMaximumTripCountInIndex())) {
    popops::mapInPlace(
        graph(),
        popops::expr::Add(popops::expr::_1, popops::expr::Const(1)),
        {iteratorTensor},
        loopContinueProg,
        debugPrefix("iterator_update"));
  }

  // 11: Add conditional around the loop body program
  loopProg.add(poplar::program::If(exitTensor, loopExitProg, loopContinueProg));

  // 12: Repeat the loop conditional program
  prog.add(poplar::program::Repeat(maxTripCountValue, loopProg));

  // 13: Copy body outputs to op outputs
  copyBodyOutputsToOpOutputs(prog);
}

namespace {
OpxCreator<LoopOpx> loopOpxCreator({Onnx::Operators::Loop_1,
                                    Onnx::Operators::Loop_11});
} // namespace
} // namespace popx
} // namespace popart
