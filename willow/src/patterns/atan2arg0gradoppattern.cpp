// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <string>
#include <popart/op/atan2.hpp>
#include <popart/patterns/atan2arg0gradoppattern.hpp>

#include "popart/ir.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

bool Atan2Arg0GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<Atan2Arg0GradOp>();
}

TensorId
Atan2Arg0GradOpPattern::makeAllReplacementOps(Op *op,
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
  auto mul     = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);

  // Connect up the new ops
  squareY->connectInTensor(0, fwdIn0.id);
  squareY->createAndConnectOutTensor(0,
                                     ir->createIntermediateTensorId(fwdIn0.id));
  squareY->setup();

  squareX->connectInTensor(0, fwdIn1.id);
  squareX->createAndConnectOutTensor(0,
                                     ir->createIntermediateTensorId(fwdIn0.id));
  squareX->setup();

  add->connectInTensor(0, squareX->outTensor(0)->id);
  add->connectInTensor(1, squareY->outTensor(0)->id);
  add->createAndConnectOutTensor(0, ir->createIntermediateTensorId(fwdIn0.id));
  add->outInfo(0) = op->prettyNpOut(add->inInfo(0), add->inInfo(1));

  div->connectInTensor(0, fwdIn1.id);
  div->connectInTensor(1, add->outTensor(0)->id);
  div->createAndConnectOutTensor(0, ir->createIntermediateTensorId(fwdIn0.id));
  div->outInfo(0) = op->prettyNpOut(div->inInfo(0), div->inInfo(1));

  mul->connectInTensor(0, gradIn.id);
  mul->connectInTensor(1, div->outTensor(0)->id);
  mul->createAndConnectOutTensor(
      0, ir->createIntermediateTensorId(div->outTensor(0)->id));
  mul->outInfo(0) = op->prettyNpOut(mul->inInfo(0), mul->inInfo(1));

  return mul->outTensor(0)->id;
}

namespace {
static PatternCreator<Atan2Arg0GradOpPattern>
    PowArg0GradOpPattern("Atan2Arg0GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
