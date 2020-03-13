// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <popart/graph.hpp>
#include <popart/op/if.hpp>
#include <popart/op/loop.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/loopx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

LoopOpx::LoopOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<LoopOp>(op, {Onnx::Operators::Loop_1, Onnx::Operators::Loop_11});
}

void LoopOpx::copyOpInputsToBodyInputs(poplar::program::Sequence &prog) const {
  auto &op              = getOp<LoopOp>();
  auto &subgraph        = op.subgraph();
  auto bodyInputs       = op.inputMap();
  auto opInputTensorIds = op.input->tensorIdMap();

  for (const auto &kv : opInputTensorIds) {
    auto opInput   = get(op.inId(kv.first));
    auto bodyInput = get(subgraph.getInputId(kv.first));
    poplar::program::Copy copyProg(opInput, bodyInput);
    prog.add(copyProg);
  }
}

// explicit body inputs: m, cond, a_in etc. offset of two so that m
// and cond is skipped
void LoopOpx::copyBodyOutputsToExplicitBodyInputs(
    poplar::program::Sequence &prog,
    const std::vector<poplar::Tensor> &bodyOutputs) const {
  auto &op                = getOp<LoopOp>();
  auto &subgraph          = op.subgraph();
  auto explicitBodyInputs = op.explicitTensorMap();

  for (int i = 0; i < bodyOutputs.size(); ++i) {
    auto bodyInputTensor  = get(subgraph.getInputId(i + 2));
    auto bodyOutputTensor = get(op.subgraph().getOutputId(i + 1));
    auto bodyOutputTemp   = cloneNcopy(prog, bodyOutputTensor);
    poplar::program::Copy copyProg(bodyOutputTemp, bodyInputTensor);
    prog.add(copyProg);
  }
}

void LoopOpx::copyBodyOutputsToOpOutputs(
    poplar::program::Sequence &prog,
    const std::vector<poplar::Tensor> &bodyOutputs) const {
  auto &op = getOp<LoopOp>();

  for (int i = 0; i < bodyOutputs.size(); ++i) {
    auto bodyOutputTensor = get(op.subgraph().getOutputId(i + 1));
    auto opOutputTensor   = get(outId(i));
    auto bodyOutputTemp   = cloneNcopy(prog, bodyOutputTensor);
    poplar::program::Copy copyProg(bodyOutputTemp, opOutputTensor);
    prog.add(copyProg);
  }
}

// Given o = builder.aiOnnxOpset10.loop([M, cond, a], 1, loopBuilder)[0]
// an offset of +1 is needed because the first variable is the boolean condition
std::vector<poplar::Tensor> LoopOpx::prepareBodyOutputs() const {
  auto &op       = getOp<LoopOp>();
  auto &subgraph = op.subgraph();
  std::vector<poplar::Tensor> bodyOutputs;

  for (int i = 0; i < subgraph.getOutputIds().size() - 1; ++i) {
    auto tensorId = subgraph.getOutputId(i + 1);
    auto tensor   = get(tensorId);
    bodyOutputs.push_back(tensor);
  }
  return bodyOutputs;
}

void LoopOpx::grow(poplar::program::Sequence &prog) const {
  auto &op         = getOp<LoopOp>();
  auto bodyOutputs = prepareBodyOutputs();

  // Set the body_output to output
  for (int i = 0; i < bodyOutputs.size(); ++i) {
    setOutTensor(i, bodyOutputs.at(i));
  }

  // 0: Define constants
  auto trueConst =
      graph().addConstant(poplar::BOOL, {}, true, debugPrefix("trueConst"));
  auto falseConst =
      graph().addConstant(poplar::BOOL, {}, false, debugPrefix("falseConst"));

  // 1: Create a poplar only iterator variable i
  auto iterTensor = graph().addVariable(poplar::INT, {}, debugPrefix("i"));
  popops::zero(graph(), iterTensor, prog, debugPrefix("zeroIterator"));

  // 2: Create a poplar only boolean variable exit
  auto exit = graph().addVariable(poplar::BOOL, {}, debugPrefix("exit"));
  poplar::program::Copy exitFalse(falseConst, exit);
  prog.add(exitFalse);

  // 3: Get user-defined max_trip_count and condition variable
  auto maxTripCountValue  = op.tripCountValue();
  auto maxTripCountTensor = getInTensor(LoopOp::getMaximumTripCountInIndex());

  auto cond   = getInTensor(LoopOp::getTerminationConditionInIndex());
  auto condIn = get(op.subgraph().getOutputId(0));

  // 4: Create loop start if prog
  poplar::program::Sequence startProg, startThenProg, startElseProg;
  copyOpInputsToBodyInputs(prog);

  auto testExpr = popops::map(
      graph(),
      popops::expr::And(popops::expr::_1,
                        popops::expr::Lt(popops::expr::_2, popops::expr::_3)),
      {condIn, iterTensor, maxTripCountTensor},
      startProg);

  // 5: Construct loop_begin_testExpr branches
  auto &loopBody = dv_p->progs.scopeFragment(op.subgraph());
  startThenProg.add(loopBody);

  auto condOut = graph().addVariable(poplar::BOOL, {}, debugPrefix("condOut"));
  poplar::program::Copy startElseCopyProg(trueConst, condOut);
  startElseProg.add(startElseCopyProg);

  startProg.add(poplar::program::If(testExpr, startThenProg, startElseProg));

  // 6: Construct loop_endTestExpr branches
  poplar::program::Sequence endProg, endThenProg, endElseProg;
  auto endTestExpr = popops::map(
      graph(),
      popops::expr::Or(popops::expr::_1,
                       popops::expr::Gte(popops::expr::_2, popops::expr::_3)),
      {condOut, iterTensor, maxTripCountTensor},
      endProg);

  poplar::program::Copy endThenCopyProg(trueConst, exit);
  endThenProg.add(endThenCopyProg);
  copyBodyOutputsToOpOutputs(endThenProg, bodyOutputs);
  copyBodyOutputsToExplicitBodyInputs(endElseProg, bodyOutputs);

  popops::mapInPlace(
      graph(),
      popops::expr::Add(popops::expr::_1, popops::expr::Const(1)),
      {iterTensor},
      endElseProg);
  endProg.add(poplar::program::If(endTestExpr, endThenProg, endElseProg));

  // 8: Construct if (!exit) {} block
  poplar::program::Sequence exitProg, exitThenProg, exitElseProg;
  exitThenProg.add(startProg);
  exitThenProg.add(endProg);

  auto exitTestExpr = popops::map(
      graph(), popops::expr::Not(popops::expr::_1), {exit}, exitProg);
  exitProg.add(poplar::program::If(exitTestExpr, exitThenProg, exitElseProg));

  // 9: Execute the loop program
  prog.add(poplar::program::Repeat(maxTripCountValue, exitProg));
}

namespace {
OpxCreator<LoopOpx> loopOpxCreator({Onnx::Operators::Loop_1,
                                    Onnx::Operators::Loop_11});
} // namespace
} // namespace popx
} // namespace popart
