// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/sigmoid.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
SigmoidOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::SigmoidInplace, 10}};
}

std::unique_ptr<Op>
SigmoidOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SigmoidInplace) {
    return std::make_unique<SigmoidInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

SigmoidInplaceOp::SigmoidInplaceOp(const SigmoidOp &sigm_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::SigmoidInplace,
                                sigm_op.getSettings()) {}

std::unique_ptr<Op> SigmoidInplaceOp::clone() const {
  return std::make_unique<SigmoidInplaceOp>(*this);
}

SigmoidOp::SigmoidOp(const OperatorIdentifier &_opid,
                     const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> SigmoidOp::clone() const {
  return std::make_unique<SigmoidOp>(*this);
}

std::vector<std::unique_ptr<Op>> SigmoidOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SigmoidGradOp>(*this));
  return upops;
}

SigmoidGradOp::SigmoidGradOp(const SigmoidOp &fwdOp)
    : Op(Onnx::GradOperators::SigmoidGrad, fwdOp.getSettings()) {}

std::unique_ptr<Op> SigmoidGradOp::clone() const {
  return std::make_unique<SigmoidGradOp>(*this);
}

const std::vector<GradInOutMapper> &SigmoidGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), SigmoidOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdOutInIndex(), SigmoidOp::getOutIndex(), GradOpInType::Out}};

  return inInfo;
}

const std::map<int, int> &SigmoidGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SigmoidOp::getInIndex()}};

  return outInfo;
}

void SigmoidGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getFwdOutInIndex());
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition sigmoidOpDef({OpDefinition::Inputs({{"X", T}}),
                                  OpDefinition::Outputs({{"Y", T}}),
                                  OpDefinition::Attributes({})});

static OpCreator<SigmoidOp> sigmoidOpCreator(OpDefinitions({
    {Onnx::Operators::Sigmoid_6, sigmoidOpDef},
}));
} // namespace

} // namespace popart
