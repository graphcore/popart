// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <utility>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/reciprocal.hpp>
#include <popart/op/square.hpp>
#include <popart/patterns/reciprocalgradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool ReciprocalGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<ReciprocalGradOp>();
}

std::vector<const Tensor *> ReciprocalGradOpPattern::touches(Op *) const {
  return {};
}

bool ReciprocalGradOpPattern::apply(Op *op) const {
  auto grad_input    = op->inTensor(0);
  auto fwd_input     = op->inTensor(1);
  auto output_tensor = op->outTensor(0);

  // create the new ops
  auto square     = makeReplacementOpInIr(Onnx::CustomOperators::Square, op);
  auto reciprocal = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Reciprocal, op);
  auto negate     = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Neg, op);
  auto mul        = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);

  // Remove the ReciprocalGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  square->connectInTensor(0, fwd_input->id);
  square->createAndConnectOutTensor(
      0, square->getIr().createIntermediateTensorId(fwd_input->id));
  square->outInfo(0) = fwd_input->info;

  reciprocal->connectInTensor(0,
                              square->outTensor(SquareOp::getOutIndex())->id);
  reciprocal->createAndConnectOutTensor(
      0, reciprocal->getIr().createIntermediateTensorId(fwd_input->id));
  reciprocal->outInfo(0) = square->outInfo(0);

  negate->connectInTensor(
      0, reciprocal->outTensor(ReciprocalOp::getOutIndex())->id);
  negate->createAndConnectOutTensor(
      0, reciprocal->getIr().createIntermediateTensorId(fwd_input->id));
  negate->outInfo(0) = reciprocal->outInfo(0);

  mul->connectInTensor(0, negate->outTensor(0)->id);
  mul->connectInTensor(1, grad_input->id);
  mul->connectOutTensor(0, output_tensor->id);

  return true;
}

namespace {
static PatternCreator<ReciprocalGradOpPattern>
    reciprocalGradOpPattern("ReciprocalGradOp",
                            /* enabled = */ true,
                            /* mandatory = */ true);

}

} // namespace popart
