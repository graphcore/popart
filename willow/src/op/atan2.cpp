// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/atan2.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

Atan2Op::Atan2Op(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseBinaryOp(_opid, settings_) {}

std::unique_ptr<Op> Atan2Op::clone() const {
  return std::make_unique<Atan2Op>(*this);
}

std::unique_ptr<Op> Atan2Op::getLhsInplaceVariant() const {
  return std::make_unique<Atan2LhsInplaceOp>(*this);
}

OperatorIdentifier Atan2Op::getLhsOperatorIdentifier() const {
  return Onnx::CustomOperators::Atan2Inplace;
}

Atan2LhsInplaceOp::Atan2LhsInplaceOp(const Atan2Op &op)
    : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::Atan2Inplace,
                                    op.getSettings()) {}

Atan2LhsInplaceOp::Atan2LhsInplaceOp(const Op::Settings &settings_)
    : ElementWiseBinaryInplaceLhsOp(Onnx::CustomOperators::Atan2Inplace,
                                    settings_) {}

std::unique_ptr<Op> Atan2LhsInplaceOp::clone() const {
  return std::make_unique<Atan2LhsInplaceOp>(*this);
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
