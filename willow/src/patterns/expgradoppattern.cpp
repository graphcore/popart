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

  auto ir   = op->pir;
  auto attr = op->nAtts.filter(sVirtualGraphAttribute);

  // create the new ops
  auto mul_op =
      make_unique<MulOp>(Onnx::Operators::Mul, ir, std::string{}, attr);

  // move ops into ir
  auto mul = mul_op.get();
  ir->moveIntoIr(std::move(mul_op));

  // Remove the ExpGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  ir->eraseOp(op->id);

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
