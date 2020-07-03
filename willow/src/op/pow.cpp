// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/pow.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

PowOp::PowOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseBinaryOp(_opid, settings_) {}

std::unique_ptr<Op> PowOp::clone() const {
  return std::make_unique<PowOp>(*this);
}

std::vector<std::unique_ptr<Op>> PowOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_in_0   = inShape(getArg0InIndex());
  const auto &shape_in_1   = inShape(getArg1InIndex());
  const auto &shape_output = outShape(getOutIndex());

  upops.emplace_back(std::make_unique<PowArg0GradOp>(
      *this, npReductionAxis(shape_in_0, shape_output)));
  upops.emplace_back(std::make_unique<PowArg1GradOp>(
      *this, npReductionAxis(shape_in_1, shape_output)));
  return upops;
}

std::unique_ptr<Op> PowOp::getLhsInplaceVariant() const {
  return std::make_unique<PowLhsInplaceOp>(*this);
}

OperatorIdentifier PowOp::getLhsOperatorIdentifier() const {
  return Onnx::CustomOperators::PowLhsInplace;
}

PowLhsInplaceOp::PowLhsInplaceOp(const PowOp &op)
    : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::PowLhsInplace,
                                    op.getSettings()) {}

PowLhsInplaceOp::PowLhsInplaceOp(const Op::Settings &settings_)
    : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::PowLhsInplace,
                                    settings_) {}

std::unique_ptr<Op> PowLhsInplaceOp::clone() const {
  return std::make_unique<PowLhsInplaceOp>(*this);
}

PowArgGradOp::PowArgGradOp(const OperatorIdentifier &_opid,
                           const std::vector<int64_t> &reduction_axes_,
                           const TensorInfo &forward_op_arg_info_,
                           const Op::Settings &settings_)
    : Op(_opid, settings_), forward_op_arg_info(forward_op_arg_info_),
      reduction_axes(reduction_axes_) {}

void PowArgGradOp::setup() { outInfo(0) = forward_op_arg_info; }

const std::vector<int64_t> &PowArgGradOp::getReductionAxes() const {
  return reduction_axes;
}

PowArg0GradOp::PowArg0GradOp(const PowOp &op,
                             const std::vector<int64_t> &reduction_axes_)
    : PowArgGradOp(Onnx::GradOperators::PowArg0Grad,
                   reduction_axes_,
                   op.inInfo(PowOp::getArg0InIndex()),
                   op.getSettings()) {}

std::unique_ptr<Op> PowArg0GradOp::clone() const {
  return std::make_unique<PowArg0GradOp>(*this);
}

const std::map<int, int> &PowArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), PowOp::getArg0InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &PowArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, getOutIndex(), GradOpInType::GradOut},
      {1, PowOp::getArg0InIndex(), GradOpInType::In},
      {2, PowOp::getArg1InIndex(), GradOpInType::In}};
  return inInfo;
}

PowArg1GradOp::PowArg1GradOp(const PowOp &op,
                             const std::vector<int64_t> &reduction_axes_)
    : PowArgGradOp(Onnx::GradOperators::PowArg1Grad,
                   reduction_axes_,
                   op.inInfo(PowOp::getArg1InIndex()),
                   op.getSettings()) {}

std::unique_ptr<Op> PowArg1GradOp::clone() const {
  return std::make_unique<PowArg1GradOp>(*this);
}

const std::map<int, int> &PowArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), PowOp::getArg1InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &PowArg1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, getOutIndex(), GradOpInType::GradOut},
      {1, PowOp::getArg0InIndex(), GradOpInType::In},
      {2, PowOp::getOutIndex(), GradOpInType::Out}};
  return inInfo;
} // namespace popart

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition powOpDef({OpDefinition::Inputs({{"X", T}, {"Y", T}}),
                              OpDefinition::Outputs({{"Z", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<PowOp> mulOpCreator(OpDefinitions(
    {{Onnx::Operators::Pow_1, powOpDef}, {Onnx::Operators::Pow_7, powOpDef}}));
} // namespace

} // namespace popart
