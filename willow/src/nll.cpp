#include <sstream>
#include <willow/error.hpp>
#include <willow/nll.hpp>
#include <willow/tensor.hpp>

namespace willow {

std::unique_ptr<Op> NllOp::clone() const {
  return std::unique_ptr<Op>(new NllOp(*this));
}

std::unique_ptr<Loss> NllLoss::clone() const {
  return std::unique_ptr<Loss>(new NllLoss(*this));
}

std::vector<std::unique_ptr<Op>> NllOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new NllGradOp(this)));
  return upops;
}

std::unique_ptr<Op> NllLoss::getOp(Ir *gp) const {
  return std::unique_ptr<Op>(
      new NllOp({op_type(), gp, {}, getWillowDomain()}, this));
}

std::string NllLoss::op_type() const { return "Nll"; }

std::vector<TensorId> NllLoss::getStreamTensorNames() const {
  return {input(labelIn())};
}

// as per pydriver.py
int NllLoss::probsIn() const { return 0; }
int NllLoss::labelIn() const { return 1; }

NllLoss::NllLoss(TensorId probs, TensorId label, TensorId output)
    : Loss({probs, label}, output) {
  // confirming that I haven't miswired things
  if (input(probsIn()) != probs || input(labelIn()) != label) {
    throw error("ILE: mis-wired tensors in calling parent constructor");
  }
}

TensorId NllLoss::probsTensorId() const { return input(probsIn()); }

TensorId NllLoss::labelTensorId() const { return input(labelIn()); }

void NllOp::setup() {
  const auto &probsInInfo = input.tensor(nlll()->probsIn())->info;
  // output is a 1-d tensor, dimension size : batchsize
  output.tensor(0)->info.set(probsInInfo.dataType(), {probsInInfo.dim(0)});
}

const NllLoss *NllOp::nlll() const { return nllloss_; }
const NllLoss *NllGradOp::nlll() const { return nllloss_; }

NllOp::NllOp(const OpConstructorBundle &b, const NllLoss *n)
    : Op(b), nllloss_(n) {}

void NllGradOp::setup() {
  // gradient of probs has same shape as probs
  auto outInfo           = input.tensor(nlll()->probsIn())->info;
  output.tensor(0)->info = outInfo;
}

NllGradOp::NllGradOp(NllOp *op_)
    : GradOp({"NllGrad", op_->pir, {}, getWillowDomain()}),
      nllloss_(op_->nlll()) {}

const std::vector<GradInOutMapper> &NllGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createNllLossGradInfo();
  return inInfo;
}

std::map<int, int> NllGradOp::createNllLossGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index probsIn()
  // the op ONLY computes the gradient of probs,
  // no gradient for label (one could interpret the
  // int as a sparse vector, but not neat)
  return {{0, nlll()->probsIn()}};
}

const std::map<int, int> &NllGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createNllLossGradOutToIn();
  return outInfo;
}

std::vector<GradInOutMapper> NllGradOp::createNllLossGradInfo() const {
  // input at index 0 : labelIn()
  // input at index 1 : probsIn()
  return {{nlll()->labelIn(), nlll()->labelIn(), GradOpInType::IN},
          {nlll()->probsIn(), nlll()->probsIn(), GradOpInType::IN}};
}

} // namespace willow
