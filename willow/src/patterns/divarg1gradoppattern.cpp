#include <memory>
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

std::vector<const Tensor *> DivArg1GradOpPattern::touches(Op *) const {
  return {};
}

// grad_out = - (grad_in * arg_0) / arg_1^2
bool DivArg1GradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(DivArg1GradOp::getGradInIndex());
  auto fwd_in0  = op->inTensor(DivArg1GradOp::getFwdArg0InIndex());
  auto fwd_in1  = op->inTensor(DivArg1GradOp::getFwdArg1InIndex());
  auto grad_out = op->outTensor(DivArg1GradOp::getOutIndex());

  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to DivArg1GradOpPattern::matches
  auto axes = dynamic_cast<DivArg1GradOp *>(op)->getReductionAxes();

  // create the new ops
  auto square = makeReplacementOpInIr(Onnx::CustomOperators::Square, op);
  auto div    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);
  auto mul    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto negate = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Neg, op);
  auto reduce = dynamic_cast<ReduceSumOp *>(
      makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::ReduceSum, op));
  reduce->setAxes(axes);
  // do not keep reduced dims
  reduce->setKeepDims(0l);

  // Remove the DivArg1GradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  square->connectInTensor(0, fwd_in1->id);
  square->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  square->outInfo(0) = square->inInfo(0);

  mul->connectInTensor(0, grad_in->id);
  mul->connectInTensor(1, fwd_in0->id);
  mul->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  mul->outInfo(0) = npOut(mul->inInfo(0), mul->inInfo(1));

  div->connectInTensor(0, mul->outTensor(0)->id);
  div->connectInTensor(1, square->outTensor(0)->id);
  div->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  div->outInfo(0) = npOut(div->inInfo(0), div->inInfo(1));

  negate->connectInTensor(0, div->outTensor(0)->id);
  negate->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  negate->outInfo(0) = negate->inInfo(0);

  reduce->connectInTensor(0, negate->outTensor(0)->id);
  reduce->connectOutTensor(0, grad_out->id);

  return true;
}

namespace {
static PatternCreator<DivArg1GradOpPattern>
    DivArg1GradOpPattern(PreAliasPatternType::DIVARG1GRADOP, "DivArg1GradOp");
}

} // namespace popart
