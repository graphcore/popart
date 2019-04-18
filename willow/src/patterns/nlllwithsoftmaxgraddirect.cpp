#include <poponnx/ir.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/patterns/nlllwithsoftmaxgraddirect.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

bool NlllWithSoftmaxGradDirect::matches(Op *op) const {

  // 1. Matches only on SoftmaxGradDirectOp
  // 2. Corresponding fwd NllLoss op must exist
  // 3. NllLoss and SoftmaxGradDirectOp must be on same IPU

  // 1.
  auto sfmgdOp = dynamic_cast<SoftmaxGradDirectOp *>(op);
  if (sfmgdOp == nullptr) {
    return false;
  }

  // 2.
  Ir &ir = op->getIr();
  if (!sfmgdOp->hasNlllFwdOp()) {
    return false;
  }

  // 3.
  auto fwdLossOp = sfmgdOp->nlllFwdOp();
  if (sfmgdOp->getVirtualGraphId() != fwdLossOp->getVirtualGraphId()) {
    return false;
  }

  return true;
}

std::vector<const Tensor *> NlllWithSoftmaxGradDirect::touches(Op *) const {
  return {};
}

bool NlllWithSoftmaxGradDirect::apply(Op *op) const {
  auto sfmgdOp   = dynamic_cast<SoftmaxGradDirectOp *>(op);
  auto fwdLossOp = sfmgdOp->nlllFwdOp();
  auto nlll      = sfmgdOp->nlll();
  Ir &ir         = fwdLossOp->getIr();

  auto label    = sfmgdOp->inTensor(nlll->getLabelInIndex());
  auto probs    = sfmgdOp->inTensor(nlll->getProbsInIndex());
  auto sfm_grad = sfmgdOp->outTensor(0);
  auto loss     = fwdLossOp->outTensor(0);

  // create the new op
  auto nlllsfmgdId = ir.moveIntoIr(std::unique_ptr<Op>(
      new NlllWithSoftmaxGradDirectOp(nlll, sfmgdOp->getSettings())));
  Op *nlllsfmgd    = ir.getOp(nlllsfmgdId);

  // Remove the SoftmaxGradDirectOp connections
  sfmgdOp->disconnectAllInputs();
  sfmgdOp->disconnectAllOutputs();

  // Remove the forward NllLossOp connections
  fwdLossOp->disconnectAllInputs();
  fwdLossOp->disconnectAllOutputs();

  // Connect up the new NlllWithSoftmaxGradDirectOp
  nlllsfmgd->connectInTensor(NlllWithSoftmaxGradDirectOp::getProbsInIndex(),
                             probs->id);
  nlllsfmgd->connectInTensor(NlllWithSoftmaxGradDirectOp::getLabelInIndex(),
                             label->id);
  nlllsfmgd->connectOutTensor(NlllWithSoftmaxGradDirectOp::getGradOutIndex(),
                              sfm_grad->id);
  nlllsfmgd->connectOutTensor(NlllWithSoftmaxGradDirectOp::getLossOutIndex(),
                              loss->id);
  nlllsfmgd->setup();

  // Remove old ops
  ir.eraseOp(sfmgdOp->id);
  ir.eraseOp(fwdLossOp->id);

  return true;
}

namespace {
static PatternCreator<NlllWithSoftmaxGradDirect>
    PreUniReplPattern(PreAliasPatternType::NLLLWITHSOFTMAXGRADDIRECT,
                      "NlllWithSoftmaxGradDirect");
}

} // namespace poponnx
