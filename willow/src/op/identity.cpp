// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/identity.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>

namespace popart {

IdentityOp::IdentityOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> IdentityOp::clone() const {
  return std::make_unique<IdentityOp>(*this);
}

std::vector<std::unique_ptr<Op>> IdentityOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<IdentityGradOp>(*this));
  return upops;
}

std::unique_ptr<Op>
IdentityOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::IdentityInplace) {
    return std::make_unique<IdentityInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}
std::vector<std::tuple<OperatorIdentifier, float>>
IdentityOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::IdentityInplace, 10}};
}

IdentityInplaceOp::IdentityInplaceOp(const IdentityOp &op)
    : IdentityOp(Onnx::CustomOperators::IdentityInplace, op.settings) {}

std::unique_ptr<Op> IdentityInplaceOp::clone() const {
  return std::make_unique<IdentityInplaceOp>(*this);
}

IdentityGradOp::IdentityGradOp(const IdentityOp &fwdOp)
    : IdentityOp(Onnx::GradOperators::IdentityGrad, fwdOp.getSettings()) {}

IdentityGradOp::IdentityGradOp(const Settings &settings_)
    : IdentityOp(Onnx::GradOperators::IdentityGrad, settings_) {}

std::unique_ptr<Op> IdentityGradOp::clone() const {
  return std::make_unique<IdentityGradOp>(*this);
}

const std::vector<GradInOutMapper> &IdentityGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), IdentityOp::getOutIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &IdentityGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), IdentityOp::getInIndex()}};

  return outInfo;
}

std::unique_ptr<Op> IdentityLossOp::clone() const {
  return std::make_unique<IdentityLossOp>(*this);
}

std::vector<std::unique_ptr<Op>> IdentityLossOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<IdentityLossGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> IdentityLoss::getOp(const Op::Settings &settings_) const {
  Op::Settings copiedSettings  = settings_;
  copiedSettings.vgraphId      = vgraphId;
  copiedSettings.pipelineStage = pipelineStage_;
  return std::unique_ptr<Op>(
      new IdentityLossOp(op_type(), this, copiedSettings));
}

const OperatorIdentifier &IdentityLoss::op_type() const {
  return Onnx::Operators::Identity_1;
}

std::vector<TensorId> IdentityLoss::getStreamTensorNames() const { return {}; }

IdentityLoss::IdentityLoss(TensorId in_, TensorId out_, ReductionType rt_)
    : Loss({in_}, out_, rt_) {}

TensorId IdentityLoss::getInputId() const { return input(0); }

const IdentityLoss *IdentityLossOp::identityl() const { return identityloss_; }
const IdentityLoss *IdentityLossGradOp::identityl() const {
  return identityloss_;
}

IdentityLossOp::IdentityLossOp(const OperatorIdentifier &_opid,
                               const IdentityLoss *n,
                               const Op::Settings &settings_)
    : LossOp(_opid, settings_), identityloss_(n) {}

void IdentityLossGradOp::setup() {
  // gradient of input has same shape as input to Id
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

void IdentityLossOp::setup() {
  if (inInfo(getInIndex()).rank() != 1) {
    throw error(
        "The identity loss {} (shape {}) is expecting a tensor of losses, 1 "
        "per sample of shape [batch_size]. Please check prior loss calculation "
        "ops to see if you have calculated the loss per-sample correctly.",
        debugName(),
        inInfo(getInIndex()).shape());
  }
  // output is a vector of length=batchsize, of the same type as input
  TensorInfo info0 = inInfo(getInIndex());
  outInfo(getOutIndex()).set(info0.dataType(), info0.shape());
}

bool IdentityLossOp::canBeReplacedByIdentity() { return true; }

IdentityLossGradOp::IdentityLossGradOp(const IdentityLossOp &op_)
    : Op(Onnx::GradOperators::IdentityLossGrad, op_.getSettings()),
      identityloss_(op_.identityl()) {}

std::unique_ptr<Op> IdentityLossGradOp::clone() const {
  return std::make_unique<IdentityLossGradOp>(*this);
}

const std::vector<GradInOutMapper> &IdentityLossGradOp::gradInputInfo() const {
  // Input at index 0 of this grad op is the input at index 0 of the identity
  // non-grad op.
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), IdentityLossOp::getInIndex(), GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &IdentityLossGradOp::gradOutToNonGradIn() const {
  // grad-op's (only) output corresponds to op's (only) input.
  static const std::map<int, int> outInfo = {
      {getOutIndex(), IdentityLossOp::getInIndex()}};
  return outInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};

// Do we support more types for this?
static OpDefinition identityOpDef({OpDefinition::Inputs({{"input", T}}),
                                   OpDefinition::Outputs({{"output", T}}),
                                   OpDefinition::Attributes({})});

static OpCreator<IdentityOp> identityOpCreator(
    OpDefinitions({{Onnx::Operators::Identity_1, identityOpDef}}));
} // namespace

} // namespace popart
