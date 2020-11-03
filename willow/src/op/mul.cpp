// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/mul.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

MulOp::MulOp(const OperatorIdentifier &_opid, const Op::Settings &_settings)
    : ElementWiseNpBroadcastableBinaryWithGradOp(_opid, _settings) {
  // TODO : Use the attributes in Mul-6
}

std::unique_ptr<Op> MulOp::clone() const {
  return std::make_unique<MulOp>(*this);
}

OperatorIdentifier MulOp::getOpId(const Ir &ir) {
  if (ir.getOpSetVersionFromModel(Domain::ai_onnx) >= 7) {
    return Onnx::Operators::Mul_7;
  } else {
    return Onnx::Operators::Mul_6;
  }
}

std::unique_ptr<Op> MulOp::getLhsInplaceVariant() const {
  return std::make_unique<MulLhsInplaceOp>(getSettings());
}

std::unique_ptr<Op> MulOp::getRhsInplaceVariant() const {
  return std::make_unique<MulRhsInplaceOp>(getSettings());
}

OperatorIdentifier MulOp::getLhsOperatorIdentifier() const {
  return Onnx::CustomOperators::MulLhsInplace;
}

OperatorIdentifier MulOp::getRhsOperatorIdentifier() const {
  return Onnx::CustomOperators::MulRhsInplace;
}

MulArg0GradOp::MulArg0GradOp(const Op &op_,
                             const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg0GradOp(Onnx::GradOperators::MulArg0Grad,
                                  _reduction_axes,
                                  op_.inInfo(MulOp::getArg0InIndex()),
                                  op_.getSettings()) {}

MulArg1GradOp::MulArg1GradOp(const Op &op,
                             const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg1GradOp(Onnx::GradOperators::MulArg1Grad,
                                  _reduction_axes,
                                  op.inInfo(MulOp::getArg1InIndex()),
                                  op.getSettings()) {}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition mulOpDef({OpDefinition::Inputs({{"A", T}, {"B", T}}),
                              OpDefinition::Outputs({{"C", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<MulOp> mulOpCreator(OpDefinitions(
    {{Onnx::Operators::Mul_6, mulOpDef}, {Onnx::Operators::Mul_7, mulOpDef}}));

} // namespace

} // namespace popart
