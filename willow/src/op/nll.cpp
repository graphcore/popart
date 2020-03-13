// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <sstream>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/nll.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::unique_ptr<Op> NllOp::clone() const {
  return std::make_unique<NllOp>(*this);
}

std::unique_ptr<Loss> NllLoss::clone() const {
  return std::make_unique<NllLoss>(*this);
}

std::vector<std::unique_ptr<Op>> NllOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<NllGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> NllLoss::getOp(const Op::Settings &settings_) const {
  Op::Settings copiedSettings  = settings_;
  copiedSettings.vgraphId      = vgraphId;
  copiedSettings.pipelineStage = pipelineStage_;
  return std::unique_ptr<Op>(
      new NllOp(Onnx::CustomOperators::Nll, *this, copiedSettings));
}

const OperatorIdentifier &NllLoss::op_type() const {
  return Onnx::CustomOperators::Nll;
}

std::vector<TensorId> NllLoss::getStreamTensorNames() const {
  return {input(getLabelInIndex())};
}

// as per pydriver.py

NllLoss::NllLoss(TensorId probs,
                 TensorId label,
                 TensorId output,
                 ReductionType rt)
    : Loss({probs, label}, output, rt) {
  // confirming that I haven't miswired things
  if (input(getProbsInIndex()) != probs || input(getLabelInIndex()) != label) {
    throw internal_error("mis-wired tensors in calling parent constructor");
  }
}

NllLoss::NllLoss(TensorId probs,
                 TensorId label,
                 TensorId output,
                 int ignoreIndex,
                 ReductionType rt)
    : NllLoss(probs, label, output, rt) {

  // An ignoreIndex has been supplied. This will influence the grow()
  // function of the loss.
  hasIgnoreIndex_ = true;
  ignoreIndex_    = ignoreIndex;
}

TensorId NllLoss::probsTensorId() const { return input(getProbsInIndex()); }

TensorId NllLoss::labelTensorId() const { return input(getLabelInIndex()); }

void NllOp::setup() {

  const auto &labelsInInfo = inInfo(nlll().getLabelInIndex());
  if (!labelsInInfo.getDataTypeInfo()->isFixedPoint()) {
    throw error(
        "Expected the label tensor NllOp to be fixed point, not the case "
        "for input with info: {}. This error for Op {}. ",
        labelsInInfo,
        str());
  }

  const auto &probsInInfo = inInfo(nlll().getProbsInIndex());
  const auto &labelInInfo = inInfo(nlll().getLabelInIndex());
  // Outputs a loss for each label index.
  // Same shape as label input, same datatype as probs input
  outInfo(nlll().getOutIndex())
      .set(probsInInfo.dataType(), labelInInfo.shape());
}

const NllLoss &NllOp::nlll() const { return nllloss_; }
const NllLoss &NllGradOp::nlll() const { return nllloss_; }

NllOp::NllOp(const OperatorIdentifier &_opid,
             const NllLoss n,
             const Op::Settings &settings_)
    : LossOp(_opid, settings_), nllloss_(n) {}

void NllOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("reduction_type",
                     static_cast<int64_t>(nlll().getReductionType()));
  os.appendAttribute("has_ignore", nlll().hasIgnoreIndex());
}

void NllGradOp::setup() {

  // connect the loss scaling tensor if is non-const
  if (!getIr().getOptimizer().lossScaling().isConst()) {
    connectInTensor(NllGradOp::getLossScalingInIndex(),
                    getIr().getOptimizer().getLossScalingTensorId(
                        inInfo(nlll().getProbsInIndex()).dataType()));
  }

  // gradient of probs has same shape as probs
  outInfo(nlll().getOutIndex()) = inInfo(nlll().getProbsInIndex());
}

NllGradOp::NllGradOp(const NllOp &op_)
    : Op(Onnx::CustomGradOperators::NllGrad, op_.getSettings()),
      nllloss_(op_.nlll()) {}

std::unique_ptr<Op> NllGradOp::clone() const {
  return std::make_unique<NllGradOp>(*this);
}

const std::vector<GradInOutMapper> &NllGradOp::gradInputInfo() const {
  // input at index 0 : labelIn()
  // input at index 1 : probsIn()
  static const std::vector<GradInOutMapper> inInfo = {
      {nlll().getLabelInIndex(), nlll().getLabelInIndex(), GradOpInType::IN},
      {nlll().getProbsInIndex(), nlll().getProbsInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &NllGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index probsIn()
  // the op ONLY computes the gradient of probs,
  // no gradient for label (one could interpret the
  // int as a sparse vector, but not neat)
  static const std::map<int, int> outInfo = {
      {getOutIndex(), nlll().getProbsInIndex()}};
  return outInfo;
}

void NllGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("reduction_type",
                     static_cast<int64_t>(nlll().getReductionType()));
  os.appendAttribute("has_ignore", nlll().hasIgnoreIndex());
}

namespace {} // namespace

} // namespace popart
