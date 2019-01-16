#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/op/reducesum.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/patterns/divarg1gradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

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

  auto ir   = op->pir;
  auto attr = op->nAtts.filter(sVirtualGraphAttribute);

  // we assume this dynamic_cast call has been confirmed
  // to be valid via a previous call to DivArg1GradOpPattern::matches
  auto axes = dynamic_cast<DivArg1GradOp *>(op)->getReductionAxes();

  // create the new ops
  auto square_op = make_unique<SquareOp>(
      Onnx::CustomOperators::Square, ir, std::string{}, attr);
  auto div_op =
      make_unique<DivOp>(Onnx::AiOnnx::OpSet9::Div, ir, std::string{}, attr);
  auto mul_op =
      make_unique<MulOp>(Onnx::AiOnnx::OpSet9::Mul, ir, std::string{}, attr);
  auto negate_op =
      make_unique<NegateOp>(Onnx::AiOnnx::OpSet9::Neg, ir, std::string{}, attr);
  auto reduce_op = make_unique<ReduceSumOp>(
      Onnx::AiOnnx::OpSet9::ReduceSum, ir, axes, false, attr);

  // move ops into ir
  auto square = square_op.get();
  auto div    = div_op.get();
  auto mul    = mul_op.get();
  auto negate = negate_op.get();
  auto reduce = reduce_op.get();
  ir->moveIntoIr(std::move(square_op));
  ir->moveIntoIr(std::move(div_op));
  ir->moveIntoIr(std::move(mul_op));
  ir->moveIntoIr(std::move(negate_op));
  ir->moveIntoIr(std::move(reduce_op));

  // Remove the DivArg1GradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  ir->eraseOp(op->id);

  // Connect up the new ops
  square->connectInTensor(0, fwd_in1->id);
  square->createAndConnectOutTensor(0, createIntermediateTensorId(grad_in->id));
  square->outInfo(0) = square->inInfo(0);

  mul->connectInTensor(0, grad_in->id);
  mul->connectInTensor(1, fwd_in0->id);
  mul->createAndConnectOutTensor(0, createIntermediateTensorId(grad_in->id));
  mul->outInfo(0) = npOut(mul->inInfo(0), mul->inInfo(1));

  div->connectInTensor(0, mul->outTensor(0)->id);
  div->connectInTensor(1, square->outTensor(0)->id);
  div->createAndConnectOutTensor(0, createIntermediateTensorId(grad_in->id));
  div->outInfo(0) = npOut(div->inInfo(0), div->inInfo(1));

  negate->connectInTensor(0, div->outTensor(0)->id);
  negate->createAndConnectOutTensor(0, createIntermediateTensorId(grad_in->id));
  negate->outInfo(0) = negate->inInfo(0);

  reduce->connectInTensor(0, negate->outTensor(0)->id);
  reduce->connectOutTensor(0, grad_out->id);

  return true;
}

namespace {
static PatternCreator<DivArg1GradOpPattern>
    DivArg1GradOpPattern(PatternType::DIVARG1GRADOP, "DivArg1GradOp");
}

} // namespace poponnx
