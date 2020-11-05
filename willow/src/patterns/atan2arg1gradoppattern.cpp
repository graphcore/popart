// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/op/add.hpp>
#include <popart/op/atan2.hpp>
#include <popart/op/div.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/op/square.hpp>
#include <popart/opidentifier.hpp>
#include <popart/patterns/atan2arg1gradoppattern.hpp>

#include <iostream>

namespace popart {

bool Atan2Arg1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<Atan2Arg1GradOp>();
}

TensorId
Atan2Arg1GradOpPattern::makeAllReplacementOps(Op *op,
                                              Ir *ir,
                                              const Tensor &gradIn,
                                              const Tensor &fwdIn0,
                                              const Tensor &fwdIn1,
                                              const Tensor &fwdOut) const {
  // create the new ops
  auto squareY = makeReplacementOpInIr(Onnx::CustomOperators::Square, op);
  auto squareX = makeReplacementOpInIr(Onnx::CustomOperators::Square, op);
  auto add     = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);
  auto div     = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);
  auto neg     = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Neg, op);

  // Connect up the new ops
  squareY->connectInTensor(0, fwdIn0.id);
  squareY->createAndConnectOutTensor(0,
                                     ir->createIntermediateTensorId(fwdIn1.id));
  squareY->setup();

  squareX->connectInTensor(0, fwdIn1.id);
  squareX->createAndConnectOutTensor(0,
                                     ir->createIntermediateTensorId(fwdIn1.id));
  squareX->setup();

  add->connectInTensor(0, squareX->outTensor(0)->id);
  add->connectInTensor(1, squareY->outTensor(0)->id);
  add->createAndConnectOutTensor(0, ir->createIntermediateTensorId(fwdIn1.id));
  add->outInfo(0) = op->prettyNpOut(add->inInfo(0), add->inInfo(1));

  div->connectInTensor(0, fwdIn0.id);
  div->connectInTensor(1, add->outTensor(0)->id);
  div->createAndConnectOutTensor(0, ir->createIntermediateTensorId(fwdIn1.id));
  div->outInfo(0) = op->prettyNpOut(div->inInfo(0), div->inInfo(1));

  neg->connectInTensor(0, div->outTensor(0)->id);
  neg->createAndConnectOutTensor(0, ir->createIntermediateTensorId(fwdIn1.id));
  neg->setup();

  return neg->outTensor(0)->id;
}

namespace {
static PatternCreator<Atan2Arg1GradOpPattern>
    PowArg0GradOpPattern(PreAliasPatternType::Atan2Arg1GradOp,
                         "Atan2Arg1GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
