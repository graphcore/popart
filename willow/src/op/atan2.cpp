// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/atan2.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

Atan2Op::Atan2Op(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseNpBroadcastableBinaryWithGradOp(_opid, settings_) {}

std::unique_ptr<Op> Atan2Op::clone() const {
  return std::make_unique<Atan2Op>(*this);
}

std::unique_ptr<Op> Atan2Op::getLhsInplaceVariant() const {
  return std::make_unique<Atan2LhsInplaceOp>(getSettings());
}

OperatorIdentifier Atan2Op::getLhsOperatorIdentifier() const {
  return Onnx::CustomOperators::Atan2Inplace;
}

std::unique_ptr<Op> Atan2LhsInplaceOp::clone() const {
  return std::make_unique<Atan2LhsInplaceOp>(*this);
}

Atan2Arg0GradOp::Atan2Arg0GradOp(const Op &op,
                                 const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg0GradOp(Onnx::GradOperators::Atan2Arg0Grad,
                                  _reduction_axes,
                                  op.inInfo(Atan2Op::getArg0InIndex()),
                                  op.getSettings()) {}

std::unique_ptr<Op> Atan2Arg0GradOp::clone() const {
  return std::make_unique<Atan2Arg0GradOp>(*this);
}

Atan2Arg1GradOp::Atan2Arg1GradOp(const Op &op,
                                 const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg1GradOp(Onnx::GradOperators::Atan2Arg1Grad,
                                  _reduction_axes,
                                  op.inInfo(Atan2Op::getArg1InIndex()),
                                  op.getSettings()) {}

std::unique_ptr<Op> Atan2Arg1GradOp::clone() const {
  return std::make_unique<Atan2Arg1GradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition atan2OpDef({OpDefinition::Inputs({{"Y", T}, {"X", T}}),
                                OpDefinition::Outputs({{"Theta", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<Atan2Op> atan2OpCreator(
    OpDefinitions({{Onnx::CustomOperators::Atan2_1, atan2OpDef}}));

} // namespace

} // namespace popart
