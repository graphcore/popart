// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/div.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/op/square.hpp>
#include <popart/patterns/divarg1gradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool DivArg1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<DivArg1GradOp>();
}

// grad_out = - (grad_in * arg_0) / arg_1^2
TensorId
DivArg1GradOpPattern::makeAllReplacementOps(Op *op,
                                            Ir *ir,
                                            const Tensor &gradIn,
                                            const Tensor &fwdIn0,
                                            const Tensor &fwdIn1,
                                            const Tensor &fwdOut) const {
  // create the new ops
  auto square = makeReplacementOpInIr(Onnx::CustomOperators::Square, op);
  auto div    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);
  auto mul    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto negate = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Neg, op);

  // Connect up the new ops
  square->connectInTensor(0, fwdIn1.id);
  square->createAndConnectOutTensor(0,
                                    ir->createIntermediateTensorId(gradIn.id));
  square->outInfo(0) = square->inInfo(0);

  mul->connectInTensor(0, gradIn.id);
  mul->connectInTensor(1, fwdIn0.id);
  mul->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  mul->outInfo(0) = op->prettyNpOut(mul->inInfo(0), mul->inInfo(1));

  div->connectInTensor(0, mul->outTensor(0)->id);
  div->connectInTensor(1, square->outTensor(0)->id);
  div->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  div->outInfo(0) = op->prettyNpOut(div->inInfo(0), div->inInfo(1));

  negate->connectInTensor(0, div->outTensor(0)->id);
  negate->createAndConnectOutTensor(0,
                                    ir->createIntermediateTensorId(gradIn.id));
  negate->outInfo(0) = negate->inInfo(0);

  return negate->outTensor(0)->id;
}

namespace {
static PatternCreator<DivArg1GradOpPattern>
    DivArg1GradOpPattern(PreAliasPatternType::DivArg1GradOp,
                         "DivArg1GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
