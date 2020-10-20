// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/expm1.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
Expm1Op::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::Expm1Inplace, 10}};
}

std::unique_ptr<Op>
Expm1Op::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::Expm1Inplace) {
    return std::make_unique<Expm1InplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

Expm1InplaceOp::Expm1InplaceOp(const Expm1Op &exp_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::Expm1Inplace,
                                exp_op.getSettings()) {}

std::unique_ptr<Op> Expm1InplaceOp::clone() const {
  return std::make_unique<Expm1InplaceOp>(*this);
}

Expm1Op::Expm1Op(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> Expm1Op::clone() const {
  return std::make_unique<Expm1Op>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition expm1OpDef({OpDefinition::Inputs({{"input", T}}),
                                OpDefinition::Outputs({{"output", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<Expm1Op> expm1OpCreator(
    OpDefinitions({{Onnx::CustomOperators::Expm1_1, expm1OpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new Expm1Op(info.opid, info.settings));
    },
    true);

} // namespace

} // namespace popart
