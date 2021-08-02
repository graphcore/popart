// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/opmanager.hpp>
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

bool IdentityLossOp::canBeReplacedByIdentity() const {
  if (getReductionType() == ReductionType::NoReduction) {
    return true;
  }

  // Scalar input
  if (inRank(getInIndex()) == 0) {
    return true;
  }

  return false;
}

IdentityLossOp::IdentityLossOp(const OperatorIdentifier &_opid,
                               const ReductionType &reduction,
                               const Op::Settings &settings_)
    : LossOp(_opid, settings_, reduction) {}

void IdentityLossOp::setup() {
  TensorInfo info0 = inInfo(getInIndex());

  Shape outShape({});

  if (getReductionType() == ReductionType::NoReduction) {
    outShape = info0.shape();
  }

  outInfo(getOutIndex()).set(info0.dataType(), outShape);
}

IdentityLossGradOp::IdentityLossGradOp(const IdentityLossOp &op_)
    : Op(Onnx::GradOperators::IdentityLossGrad, op_.getSettings()),
      reduction_type_(op_.getReductionType()),
      outShape_(op_.inShape(IdentityOp::getInIndex())) {}

void IdentityLossGradOp::setup() {
  // gradient of input has same shape as input to Id
  outInfo(getOutIndex()).set(inInfo(getInIndex()).dataType(), outShape_);
}

bool IdentityLossGradOp::canBeReplacedByIdentity() const {
  return getReductionType() == ReductionType::NoReduction;
}

std::unique_ptr<Op> IdentityLossGradOp::clone() const {
  return std::make_unique<IdentityLossGradOp>(*this);
}

const std::vector<GradInOutMapper> &IdentityLossGradOp::gradInputInfo() const {
  // Input at index 0 of this grad op is the input at index 0 of the identity
  // non-grad op.
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), IdentityLossOp::getOutIndex(), GradOpInType::GradOut}};
  return inInfo;
}

const std::map<int, int> &IdentityLossGradOp::gradOutToNonGradIn() const {
  // grad-op's (only) output corresponds to op's (only) input.
  static const std::map<int, int> outInfo = {
      {getOutIndex(), IdentityLossOp::getInIndex()}};
  return outInfo;
}

float IdentityLossGradOp::getShardRescaleFactor(Op *const shardedOp,
                                                OutIndex index) const {
  if (reduction_type_ == ReductionType::Mean) {
    return static_cast<float>(shardedOp->outInfo(index).nelms()) /
           static_cast<float>(outInfo(index).nelms());
  }
  return Op::getShardRescaleFactor(shardedOp, index);
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
static OpDefinition
    identityOpDef({OpDefinition::Inputs({{"input", T}}),
                   OpDefinition::Outputs({{"output", T}}),
                   OpDefinition::Attributes({{"reduction", {"*"}}})});

static OpCreator<IdentityOp> identityOpCreator(
    OpDefinitions({{Onnx::Operators::Identity_1, identityOpDef}}));
static OpCreator<IdentityLossOp> identityLossOpCreator(
    OpDefinitions({{Onnx::CustomOperators::IdentityLoss, identityOpDef}}),
    [](const OpCreatorInfo &info) {
      std::string reductionStr =
          info.attributes.getAttribute<Attributes::String>("reduction");
      ReductionType reduction = LossOp::reductionTypeFromString(reductionStr);
      return std::unique_ptr<IdentityLossOp>(
          new IdentityLossOp(info.opid, reduction, info.settings));
    },
    true);
} // namespace

} // namespace popart
