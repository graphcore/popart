// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/op/elementwise.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/patterns/binarygradoppattern.hpp>

namespace popart {

std::vector<const Tensor *> BinaryGradOpPattern::touches(Op *) const {
  return {};
}

bool BinaryGradOpPattern::apply(Op *op) const {
  // Save all tensors prior to disconnection
  auto grad_in  = op->inTensor(ElementWiseBinaryGradOp::getGradInIndex());
  auto fwd_in0  = op->inTensor(ElementWiseBinaryGradOp::getFwdArg0InIndex());
  auto fwd_in1  = op->inTensor(ElementWiseBinaryGradOp::getFwdArg1InIndex());
  auto fwd_out  = op->inTensor(ElementWiseBinaryGradOp::getFwdOutIndex());
  auto grad_out = op->outTensor(ElementWiseBinaryGradOp::getOutIndex());

  // Disconnect the original op
  op->disconnectAllInputs();
  op->disconnectAllOutputs();

  // we assume this dynamic_cast call has been confirmed to be valid via a
  // previous call to matches which is a sub class of ElementWiseBinaryGradOp
  auto axes = dynamic_cast<ElementWiseBinaryGradOp *>(op)->getReductionAxes();

  // Create the reduce op
  auto reduce = dynamic_cast<ReduceSumOp *>(
      makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::ReduceSum, op));
  reduce->setAxes(axes);
  // do not keep reduced dims
  reduce->setKeepDims(0l);

  TensorId pre_reduce = makeAllReplacementOps(
      op, &grad_in->getIr(), *grad_in, *fwd_in0, *fwd_in1, *fwd_out);

  reduce->connectInTensor(0, pre_reduce);

  reduce->connectOutTensor(0, grad_out->id);

  // Don't delete op until after the op->prettyNpOut calls.
  op->getGraph().eraseOp(op->id);

  return true;
}

} // namespace popart
