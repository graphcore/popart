#include <neuralnet/error.hpp>
#include <neuralnet/nll.hpp>
#include <neuralnet/tensor.hpp>
#include <sstream>

namespace neuralnet {

std::vector<std::unique_ptr<Op>> NllOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new NllGradOp(this)));
  return upops;
}

std::unique_ptr<Op> NllLoss::getOp(Graph *gp) const {
  return std::unique_ptr<Op>(
      new NllOp({op_type(), gp, {}, getNeuralNetDomain()}, this));
}

std::string NllLoss::op_type() const { return "Nll"; }

std::vector<TensorId> NllLoss::getStreamTensorNames() const {
  return {input(labelsIn())};
}

NllLoss::NllLoss(const std::string &argstring) : Loss(argstring) {
  // expect 2 inputs, 0 args.
  confirmSizes(2, 0);
}

void NllOp::setup() {
  // output is a scalar of the same type as probs
  //

  output.tensor(0)->info.set(input.tensor(nlll()->probsIn())->info.dataType(),
                             {});
}

const NllLoss *NllOp::nlll() const { return nllloss_; }

NllOp::NllOp(const OpConstructorBundle &b, const NllLoss *n)
    : Op(b), nllloss_(n) {}

void NllGradOp::setup() {
  // gradient of probs has same shape as probs

  std::stringstream ss;
  auto outInfo = nlllossOp->input.tensor(nlllossOp->nlll()->probsIn())->info;
  output.tensor(0)->info = outInfo;
  outInfo.append(ss);
  std::cout << ss.str() << std::endl;
}

NllGradOp::NllGradOp(NllOp *op_)
    : GradOp({"NllGrad", op_->pgraph, {}, getNeuralNetDomain()}),
      nlllossOp(op_) {}

Op *NllGradOp::getNonGradOp() const { return nlllossOp; }

const std::vector<GradInOutMapper> &NllGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createNllLossGradInfo();
  return inInfo;
}

std::map<int, int> NllGradOp::createNllLossGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index probsIn()
  // the op ONLY computes the gradient of probs,
  // no gradient for labels (one could interpret the
  // int as a sparse vector, but not neat)
  return {{0, nlllossOp->nlll()->probsIn()}};
}

// as per pydriver.py
int NllLoss::probsIn() const { return 0; }
int NllLoss::labelsIn() const { return 1; }

const std::map<int, int> &NllGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createNllLossGradOutToIn();
  return outInfo;
}

std::vector<GradInOutMapper> NllGradOp::createNllLossGradInfo() const {
  // input at index 0 : labelsIn()
  // input at index 1 : probsIn()
  return {{0, nlllossOp->nlll()->labelsIn(), GradOpInType::IN},
          {1, nlllossOp->nlll()->probsIn(), GradOpInType::IN}};
}

} // namespace neuralnet
