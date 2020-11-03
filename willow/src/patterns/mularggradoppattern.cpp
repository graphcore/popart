// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/mul.hpp>
#include <popart/patterns/mularggradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensors.hpp>

namespace popart {

bool MulArgGradOpPattern::matches(Op *op) const {
  if (op->isConvertibleTo<MulArg0GradOp>()) {
    return true;
  }
  if (op->isConvertibleTo<MulArg1GradOp>()) {
    return true;
  }
  return false;
}

std::vector<const Tensor *> MulArgGradOpPattern::touches(Op *) const {
  return {};
}

bool MulArgGradOpPattern::apply(Op *op) const {
  auto grad_in = op->inTensor(ElementWiseBinaryGradOp::getGradInIndex());

  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to MulArgGradOpPattern::matches

  Tensor *other_fwd_in =
      op->isConvertibleTo<MulArg0GradOp>()
          ? op->inTensor(ElementWiseBinaryGradOp::getFwdArg1InIndex())
          : op->inTensor(ElementWiseBinaryGradOp::getFwdArg0InIndex());
  auto grad_out = op->outTensor(ElementWiseBinaryGradOp::getOutIndex());

  auto axes = dynamic_cast<ElementWiseBinaryGradOp *>(op)->getReductionAxes();

  // create the new ops
  auto mul    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto reduce = dynamic_cast<ReduceSumOp *>(
      makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::ReduceSum, op));
  reduce->setAxes(axes);
  // do not keep reduced dims
  reduce->setKeepDims(0l);

  // Disconnect the original op
  op->disconnectAllInputs();
  op->disconnectAllOutputs();

  // Connect up the new ops
  mul->connectInTensor(0, grad_in->id);
  mul->connectInTensor(1, other_fwd_in->id);
  mul->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  mul->outInfo(0) = op->prettyNpOut(mul->inInfo(0), mul->inInfo(1));

  reduce->connectInTensor(0, mul->outTensor(0)->id);
  reduce->connectOutTensor(0, grad_out->id);

  // Eras the original op
  op->getGraph().eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<MulArgGradOpPattern>
    MulArgGradOpPattern(PreAliasPatternType::MulArgGradOp,
                        "MulArgGradOp",
                        /* enabled = */ true,
                        /* mandatory = */ true);
}

} // namespace popart
