#include <poponnx/ir.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/patterns/softmaxgraddirect.hpp>

namespace poponnx {

// NLLGRAD (0) -> x -> SOFTMAXGRAD.
const OperatorIdentifier &SoftmaxGradDirect::get0() const {
  return Onnx::CustomGradOperators::NllGrad;
}

// NLLGRAD -> x -> SOFTMAXGRAD (1).
const OperatorIdentifier &SoftmaxGradDirect::get1() const {
  return Onnx::GradOperators::SoftmaxGrad;
}

OpId SoftmaxGradDirect::moveMergedIntoIr(Op *opRoot) const {
  // The root of the pattern is an NLLGrad,
  // we need to move from it th the SoftmaxOp
  Ir *pir     = opRoot->pir;
  Op *nllgrad = opRoot;

  return pir->moveIntoIr(std::unique_ptr<Op>(new SoftmaxGradDirectOp(
      pir, dynamic_cast<NllGradOp *>(nllgrad)->nlll())));
}

namespace {
static PatternCreator<SoftmaxGradDirect>
    PreUniReplPattern(PatternType::SOFTMAXGRADDIRECT, "SoftmaxGradDirect");
}

} // namespace poponnx
