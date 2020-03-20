// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/abs.hpp>
#include <popart/opmanager.hpp>

namespace popart {

AbsOp::AbsOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> AbsOp::clone() const {
  return std::make_unique<AbsOp>(*this);
}

std::vector<std::unique_ptr<Op>> AbsOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<AbsGradOp>(*this));
  return upops;
}

AbsGradOp::AbsGradOp(const AbsOp &op_)
    : Op(Onnx::GradOperators::AbsGrad, op_.getSettings()) {}

const std::map<int, int> &AbsGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AbsOp::getInIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AbsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), AbsOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdArgInIndex(), AbsOp::getInIndex(), GradOpInType::IN}};
  return inInfo;
}

void AbsGradOp::setup() { outInfo(getOutIndex()) = inInfo(getGradInIndex()); }

std::unique_ptr<Op> AbsGradOp::clone() const {
  return std::make_unique<AbsGradOp>(*this);
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
                                    DataType::FLOAT};

static OpCreator<AbsOp>
    absOpCreator(OpDefinitions({{Onnx::Operators::Abs_6,
                                 {OpDefinition::Inputs({{"x", T}}),
                                  OpDefinition::Outputs({{"y", T}}),
                                  OpDefinition::Attributes({})}}}),
                 true);
} // namespace

} // namespace popart
