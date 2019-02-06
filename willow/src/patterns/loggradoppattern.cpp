#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/op/log.hpp>
#include <poponnx/patterns/loggradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool LogGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<LogGradOp>();
}

std::vector<const Tensor *> LogGradOpPattern::touches(Op *) const { return {}; }

// grad_out = grad_in / fwd_in
bool LogGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(LogGradOp::getGradInIndex());
  auto fwd_in   = op->inTensor(LogGradOp::getFwdArgInIndex());
  auto grad_out = op->outTensor(LogGradOp::getOutIndex());

  // create the new ops
  auto div = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);

  // Remove the LogGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getIr().eraseOp(op->id);

  // Connect up the new ops
  div->connectInTensor(DivOp::getArg0InIndex(), grad_in->id);
  div->connectInTensor(DivOp::getArg1InIndex(), fwd_in->id);
  div->connectOutTensor(DivOp::getOutIndex(), grad_out->id);
  div->setup();

  return true;
}

namespace {
static PatternCreator<LogGradOpPattern>
    LogGradOpPattern(PreAliasPatternType::LOGGRADOP, "LogGradOp");
}

} // namespace poponnx
