#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/exp.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/patterns/expgradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool ExpGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<ExpGradOp>();
}

std::vector<const Tensor *> ExpGradOpPattern::touches(Op *) const { return {}; }

// grad_out = grad_in * fwd_out
bool ExpGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(ExpGradOp::getGradInIndex());
  auto fwd_out  = op->inTensor(ExpGradOp::getFwdOutInIndex());
  auto grad_out = op->outTensor(ExpGradOp::getOutIndex());

  // create the new ops
  auto mul = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);

  // Remove the ExpGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->pir->eraseOp(op->id);

  // Connect up the new ops
  mul->connectInTensor(MulOp::getArg0InIndex(), grad_in->id);
  mul->connectInTensor(MulOp::getArg1InIndex(), fwd_out->id);
  mul->connectOutTensor(MulOp::getOutIndex(), grad_out->id);

  return true;
}

namespace {
static PatternCreator<ExpGradOpPattern> ExpGradOpPattern(PatternType::EXPGRADOP,
                                                         "ExpGradOp");
}

} // namespace poponnx
