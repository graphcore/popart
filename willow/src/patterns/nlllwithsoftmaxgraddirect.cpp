#include <popart/graph.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/softmax.hpp>
#include <popart/patterns/nlllwithsoftmaxgraddirect.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

namespace popart {

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
  if (!sfmgdOp->hasNlllFwdOp()) {
    return false;
  }

  // 3.
  auto fwdLossOp = sfmgdOp->nlllFwdOp();
  if (sfmgdOp->getOptionalVirtualGraphId() !=
      fwdLossOp->getOptionalVirtualGraphId()) {
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
  auto &nlll     = sfmgdOp->nlll();
  auto &graph    = op->getGraph();

  auto label    = sfmgdOp->inTensor(NllLoss::getLabelInIndex());
  auto probs    = sfmgdOp->inTensor(NllLoss::getProbsInIndex());
  auto sfm_grad = sfmgdOp->outTensor(0);
  auto loss     = fwdLossOp->outTensor(0);

  // create the new op
  auto nlllsfmgdId = graph.moveIntoGraph(std::unique_ptr<Op>(
      new NlllWithSoftmaxGradDirectOp(nlll, sfmgdOp->getSettings())));
  Op *nlllsfmgd    = graph.getOp(nlllsfmgdId);

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
  sfmgdOp->getGraph().eraseOp(sfmgdOp->id);
  fwdLossOp->getGraph().eraseOp(fwdLossOp->id);

  return true;
}

namespace {
static PatternCreator<NlllWithSoftmaxGradDirect>
    PreUniReplPattern(PreAliasPatternType::NLLLWITHSOFTMAXGRADDIRECT,
                      "NlllWithSoftmaxGradDirect");
}

} // namespace popart
