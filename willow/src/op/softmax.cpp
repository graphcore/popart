// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/softmax.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

namespace popart {

SoftmaxOp::SoftmaxOp(const OperatorIdentifier &_opid,
                     int64_t axis_,
                     const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_), axis(axis_) {}

SoftmaxInplaceOp::SoftmaxInplaceOp(const SoftmaxOp &softmax_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::SoftmaxInplace,
                                softmax_op.getSettings()),
      axis(softmax_op.getAxis()) {}

std::vector<std::unique_ptr<Op>> SoftmaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SoftmaxGradOp>(*this));
  return upops;
}

std::vector<std::tuple<OperatorIdentifier, float>>
SoftmaxOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::SoftmaxInplace, 10}};
}

std::unique_ptr<Op> SoftmaxInplaceOp::clone() const {
  return std::make_unique<SoftmaxInplaceOp>(*this);
}

std::unique_ptr<Op>
SoftmaxOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SoftmaxInplace) {
    return std::make_unique<SoftmaxInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

std::unique_ptr<Op> SoftmaxOp::clone() const {
  return std::make_unique<SoftmaxOp>(*this);
}

int64_t SoftmaxOp::getAxis() const {
  if (axis < 0) {
    return inRank(getInIndex()) + axis;
  } else {
    return axis;
  }
}

void SoftmaxOp::setAxis(int64_t axis_) { axis = axis_; }

void SoftmaxOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

void SoftmaxInplaceOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

void SoftmaxGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradProbsInIndex());
}

SoftmaxGradOp::SoftmaxGradOp(const SoftmaxOp &op_)
    : Op(Onnx::GradOperators::SoftmaxGrad, op_.getSettings()),
      axis(op_.getAxis()) {}

std::unique_ptr<Op> SoftmaxGradOp::clone() const {
  return std::make_unique<SoftmaxGradOp>(*this);
}

const std::vector<GradInOutMapper> &SoftmaxGradOp::gradInputInfo() const {
  // input at index 0 (probGradInputIndex()) : gradient of output of softmax
  // input at index 1 (actsIn()): input of softmax (activations before p)
  // the (1-sparse) gradient of the output will be used
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradProbsInIndex(), SoftmaxOp::getOutIndex(), GradOpInType::GradOut},
      {getActsInIndex(), SoftmaxOp::getInIndex(), GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &SoftmaxGradOp::gradOutToNonGradIn() const {
  // the grad-op's output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SoftmaxOp::getInIndex()}};
  return outInfo;
}

int64_t SoftmaxGradOp::getAxis() const { return axis; }

void SoftmaxGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

SoftmaxGradDirectOp::SoftmaxGradDirectOp(
    const TensorId lossId,
    const nonstd::optional<int> ignoreIndex,
    const ReductionType reduction,
    const Op::Settings &_settings)
    : Op(Onnx::CustomGradOperators::SoftmaxGradDirect, _settings),
      lossId_(lossId), reduction_(reduction), ignoreIndex_(ignoreIndex) {}

std::unique_ptr<Op> SoftmaxGradDirectOp::clone() const {
  throw error("Unexpected (but valid) request to clone SoftmaxGradDirectOp");
}

void SoftmaxGradDirectOp::setup() {
  // gradient of activations has same shape as probabilities
  outInfo(getOutIndex()) = inInfo(getProbsInIndex());
}

bool SoftmaxGradDirectOp::hasNlllFwdOp() const {
  if (getGraph().getTensors().contains(lossId_)) {
    auto t = getGraph().getTensors().get(lossId_);
    return t->hasProducer();
  }
  return false;
}

Op *SoftmaxGradDirectOp::nlllFwdOp() const {

  // First check that the forward Nll loss op exists
  // in the ir
  if (!hasNlllFwdOp()) {
    throw error("The forward loss op corresponding to the SoftmaxGradDirectOp "
                "{} does not exist in the Ir",
                id);
  }

  // Find the op producing the loss tensor, i.e. the
  // corresponding fwd loss op whose bwd op has merged
  // with the SoftmaxGradOp
  Tensor *lossTensor = getGraph().getTensors().get(lossId_);
  Op *fwdLossOp      = lossTensor->getProducer();

  return fwdLossOp;
}

void SoftmaxGradDirectOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("reduction_type", static_cast<int64_t>(reduction_));
  if (hasIgnoreIndex()) {
    os.appendAttribute("ignore_index", static_cast<int64_t>(*ignoreIndex_));
  }
}

NlllWithSoftmaxGradDirectOp::NlllWithSoftmaxGradDirectOp(
    const nonstd::optional<int> ignoreIndex,
    const ReductionType reduction,
    const Op::Settings &_settings)
    : Op(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect, _settings),
      reduction_(reduction), ignoreIndex_(ignoreIndex) {}

std::unique_ptr<Op> NlllWithSoftmaxGradDirectOp::clone() const {
  return std::make_unique<NlllWithSoftmaxGradDirectOp>(*this);
}

void NlllWithSoftmaxGradDirectOp::setup() {

  // gradient of activations has same shape as probabilities
  outInfo(getGradOutIndex()) = inInfo(getProbsInIndex());

  Shape outShape({});
  if (getReductionType() == ReductionType::NoReduction) {
    outShape = inInfo(getLabelInIndex()).shape();
  }

  outInfo(getLossOutIndex())
      .set(inInfo(getProbsInIndex()).dataType(), outShape);
}

void NlllWithSoftmaxGradDirectOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("reduction_type", static_cast<int64_t>(reduction_));
  if (hasIgnoreIndex()) {
    os.appendAttribute("ignore_index", static_cast<int64_t>(*ignoreIndex_));
  }
}

float NlllWithSoftmaxGradDirectOp::getShardRescaleFactor(Op *const shardedOp,
                                                         OutIndex index) const {
  if (reduction_ == ReductionType::Mean && index == getGradOutIndex()) {
    return static_cast<float>(shardedOp->inInfo(getProbsInIndex()).nelms()) /
           static_cast<float>(inInfo(getProbsInIndex()).nelms());
  }
  return Op::getShardRescaleFactor(shardedOp, index);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition softmaxOpDef({OpDefinition::Inputs({{"input", T}}),
                                  OpDefinition::Outputs({{"output", T}}),
                                  OpDefinition::Attributes({{"axis", {"*"}}})});

static OpCreator<SoftmaxOp> softmaxOpCreator(
    OpDefinitions({{Onnx::Operators::Softmax_1, softmaxOpDef},
                   {Onnx::Operators::Softmax_11, softmaxOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t axis = info.attributes.getAttribute<Attributes::Int>("axis", 1);

      return std::unique_ptr<Op>(new SoftmaxOp(info.opid, axis, info.settings));
    },
    true);

} // namespace

} // namespace popart
