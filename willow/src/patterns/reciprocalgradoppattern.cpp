#include <utility>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/op/reciprocal.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/patterns/reciprocalgradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

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
  auto ir            = op->pir;
  auto attr          = op->nAtts.filter(sVirtualGraphAttribute);

  // create the new ops
  auto square_op = make_unique<SquareOp>(
      Onnx::CustomOperators::Square, ir, std::string{}, attr);
  auto reciprocal_op = make_unique<ReciprocalOp>(
      Onnx::AiOnnx::OpSet9::Reciprocal, ir, std::string{}, attr);
  auto negate_op =
      make_unique<NegateOp>(Onnx::AiOnnx::OpSet9::Neg, ir, std::string{}, attr);
  auto mul_op =
      make_unique<MulOp>(Onnx::AiOnnx::OpSet9::Mul, ir, std::string{}, attr);

  // move ops into ir
  auto square     = square_op.get();
  auto reciprocal = reciprocal_op.get();
  auto negate     = negate_op.get();
  auto mul        = mul_op.get();
  ir->moveIntoIr(std::move(square_op));
  ir->moveIntoIr(std::move(reciprocal_op));
  ir->moveIntoIr(std::move(negate_op));
  ir->moveIntoIr(std::move(mul_op));

  // Remove the ReciprocalGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  ir->eraseOp(op->id);

  // Connect up the new ops
  square->connectInTensor(0, fwd_input->id);
  square->createAndConnectOutTensor(0,
                                    createIntermediateTensorId(fwd_input->id));
  square->outInfo(0) = fwd_input->info;

  reciprocal->connectInTensor(0,
                              square->outTensor(SquareOp::getOutIndex())->id);
  reciprocal->createAndConnectOutTensor(
      0, createIntermediateTensorId(fwd_input->id));
  reciprocal->outInfo(0) = square->outInfo(0);

  negate->connectInTensor(
      0, reciprocal->outTensor(ReciprocalOp::getOutIndex())->id);
  negate->createAndConnectOutTensor(0,
                                    createIntermediateTensorId(fwd_input->id));
  negate->outInfo(0) = reciprocal->outInfo(0);

  mul->connectInTensor(0, negate->outTensor(0)->id);
  mul->connectInTensor(1, grad_input->id);
  mul->connectOutTensor(0, output_tensor->id);

  return true;
}

namespace {
static PatternCreator<ReciprocalGradOpPattern>
    reciprocalGradOpPattern(PatternType::RECIPROCALGRADOP, "ReciprocalGradOp");
}

} // namespace poponnx
