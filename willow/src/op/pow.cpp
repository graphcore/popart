// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/pow.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

PowOp::PowOp(const OperatorIdentifier &_opid, const Op::Settings &_settings)
    : ElementWiseNpBroadcastableBinaryWithGradOp(_opid, _settings) {}

std::unique_ptr<Op> PowOp::clone() const {
  return std::make_unique<PowOp>(*this);
}

std::unique_ptr<Op> PowOp::getLhsInplaceVariant() const {
  return std::make_unique<PowLhsInplaceOp>(getSettings());
}

OperatorIdentifier PowOp::getLhsOperatorIdentifier() const {
  return Onnx::CustomOperators::PowLhsInplace;
}

PowArg0GradOp::PowArg0GradOp(const Op &op,
                             const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg0GradOp(Onnx::GradOperators::PowArg0Grad,
                                  _reduction_axes,
                                  op.inInfo(PowOp::getArg0InIndex()),
                                  op.getSettings()) {}

PowArg1GradOp::PowArg1GradOp(const Op &op,
                             const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg1GradOp(Onnx::GradOperators::PowArg1Grad,
                                  _reduction_axes,
                                  op.inInfo(PowOp::getArg1InIndex()),
                                  op.getSettings()) {}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition powOpDef({OpDefinition::Inputs({{"X", T}, {"Y", T}}),
                              OpDefinition::Outputs({{"Z", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<PowOp> mulOpCreator(OpDefinitions(
    {{Onnx::Operators::Pow_1, powOpDef}, {Onnx::Operators::Pow_7, powOpDef}}));
} // namespace

} // namespace popart
