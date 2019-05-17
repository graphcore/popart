#include <poponnx/attributes.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/dropout.hpp>
#include <poponnx/patterns/dropoutgradoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool DropoutGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<DropoutGradOp>();
}

std::vector<const Tensor *> DropoutGradOpPattern::touches(Op *) const {
  return {};
}

// grad_out = grad_in / fwd_in
bool DropoutGradOpPattern::apply(Op *op) const {
  auto drGrad = dynamic_cast<DropoutGradOp *>(op);

  auto grad_in  = drGrad->inTensor(DropoutGradOp::getGradInIndex());
  auto grad_out = drGrad->outTensor(DropoutGradOp::getOutIndex());

  // Create the new op
  auto newOp   = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Dropout, drGrad);
  auto dropout = dynamic_cast<DropoutOp *>(newOp);
  dropout->setSeedModifier(drGrad->getSeedModifier());
  dropout->setRatio(drGrad->getRatio());

  // Remove the DropoutGradOp
  drGrad->disconnectAllInputs();
  drGrad->disconnectAllOutputs();
  drGrad->getGraph().eraseOp(drGrad->id);

  // Connect up the new ops
  dropout->connectInTensor(DropoutOp::getInIndex(), grad_in->id);
  dropout->connectOutTensor(DropoutOp::getOutIndex(), grad_out->id);
  dropout->setup();

  return true;
}

namespace {
static PatternCreator<DropoutGradOpPattern>
    DropoutGradOpPattern(PreAliasPatternType::DROPOUTGRADOP, "DropoutGradOp");
}

} // namespace poponnx
