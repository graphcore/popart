#include <sstream>
#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

std::unique_ptr<Op> NllOp::clone() const { return make_unique<NllOp>(*this); }

std::unique_ptr<Loss> NllLoss::clone() const {
  return make_unique<NllLoss>(*this);
}

std::vector<std::unique_ptr<Op>> NllOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<NllGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> NllLoss::getOp(const Op::Settings &settings_) const {
  Op::Settings copiedSettings = settings_;
  copiedSettings.vgraphId     = vgraphId;
  return std::unique_ptr<Op>(
      new NllOp(Onnx::CustomOperators::Nll, this, copiedSettings));
}

const OperatorIdentifier &NllLoss::op_type() const {
  return Onnx::CustomOperators::Nll;
}

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

  const auto &labelsInInfo = inInfo(nlll()->getLabelInIndex());
  if (!labelsInInfo.getDataTypeInfo()->isFixedPoint()) {
    throw error(
        "Expected the label tensor NllOp to be fixed point, not the case "
        "for input with info: {}. This error for Op {}. ",
        labelsInInfo,
        str());
  }

  const auto &probsInInfo = inInfo(nlll()->getProbsInIndex());
  // output is a 1-d tensor, dimension size : batchsize
  outInfo(0).set(probsInInfo.dataType(), {probsInInfo.dim(0)});
}

const NllLoss *NllOp::nlll() const { return nllloss_; }
const NllLoss *NllGradOp::nlll() const { return nllloss_; }

NllOp::NllOp(const OperatorIdentifier &_opid,
             const NllLoss *n,
             const Op::Settings &settings_)
    : LossOp(_opid, settings_), nllloss_(n) {}

void NllGradOp::setup() {
  // gradient of probs has same shape as probs
  auto out_info = inInfo(nlll()->getProbsInIndex());
  outInfo(0)    = out_info;
}

NllGradOp::NllGradOp(const NllOp &op_)
    : Op(Onnx::CustomGradOperators::NllGrad, op_.getSettings()),
      nllloss_(op_.nlll()) {}

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

namespace {} // namespace

} // namespace poponnx
