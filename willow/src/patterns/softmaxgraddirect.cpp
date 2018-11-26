#include <poponnx/ir.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/patterns/softmaxgraddirect.hpp>

namespace willow {

// NLLGRAD (0) -> x -> SOFTMAXGRAD.
OpType SoftmaxGradDirect::get0() const { return OpType::NLLGRAD; }

// NLLGRAD -> x -> SOFTMAXGRAD (1).
OpType SoftmaxGradDirect::get1() const { return OpType::SOFTMAXGRAD; }

OpId SoftmaxGradDirect::moveMergedIntoIr(Op *opRoot) const {
  // The root of the pattern is an NLLGrad,
  // we need to move from it th the SoftmaxOp
  Ir *pir     = opRoot->pir;
  Op *nllgrad = opRoot;

  return pir->moveIntoIr(std::unique_ptr<Op>(new SoftmaxGradDirectOp(
      pir, dynamic_cast<NllGradOp *>(nllgrad)->nlll())));
}

} // namespace willow
