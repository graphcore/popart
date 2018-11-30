#include <sstream>
#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

std::unique_ptr<Op> NllOp::clone() const { return make_unique<NllOp>(*this); }

std::unique_ptr<Loss> NllLoss::clone() const {
  return make_unique<NllLoss>(*this);
}

std::vector<std::unique_ptr<Op>> NllOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<NllGradOp>(this));
  return upops;
}

std::unique_ptr<Op> NllLoss::getOp(Ir *gp) const {
  return std::unique_ptr<Op>(
      new NllOp({op_type(), gp, {}, getPoponnxDomain()}, this));
}

std::string NllLoss::op_type() const { return "Nll"; }

std::vector<TensorId> NllLoss::getStreamTensorNames() const {
  return {input(getLabelInIndex())};
}

// as per pydriver.py

NllLoss::NllLoss(TensorId probs, TensorId label, TensorId output)
    : Loss({probs, label}, output) {
  // confirming that I haven't miswired things
  if (input(getProbsInIndex()) != probs || input(getLabelInIndex()) != label) {
    throw error("ILE: mis-wired tensors in calling parent constructor");
  }
}

TensorId NllLoss::probsTensorId() const { return input(getProbsInIndex()); }

TensorId NllLoss::labelTensorId() const { return input(getLabelInIndex()); }

void NllOp::setup() {
  const auto &probsInInfo = inInfo(nlll()->getProbsInIndex());
  // output is a 1-d tensor, dimension size : batchsize
  outInfo(0).set(probsInInfo.dataType(), {probsInInfo.dim(0)});
}

const NllLoss *NllOp::nlll() const { return nllloss_; }
const NllLoss *NllGradOp::nlll() const { return nllloss_; }

NllOp::NllOp(const OpConstructorBundle &b, const NllLoss *n)
    : LossOp(b), nllloss_(n) {}

void NllGradOp::setup() {
  // gradient of probs has same shape as probs
  auto out_info = inInfo(nlll()->getProbsInIndex());
  outInfo(0)    = out_info;
}

NllGradOp::NllGradOp(NllOp *op_)
    : Op({"NllGrad", op_->pir, {}, getPoponnxDomain()}), nllloss_(op_->nlll()) {
}

const std::vector<GradInOutMapper> &NllGradOp::gradInputInfo() const {
  // input at index 0 : labelIn()
  // input at index 1 : probsIn()
  static const std::vector<GradInOutMapper> inInfo = {
      {nlll()->getLabelInIndex(), nlll()->getLabelInIndex(), GradOpInType::IN},
      {nlll()->getProbsInIndex(), nlll()->getProbsInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &NllGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index probsIn()
  // the op ONLY computes the gradient of probs,
  // no gradient for label (one could interpret the
  // int as a sparse vector, but not neat)
  static const std::map<int, int> outInfo = {
      {getOutIndex(), nlll()->getProbsInIndex()}};
  return outInfo;
}

} // namespace poponnx
