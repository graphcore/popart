// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/ceil.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/onewayunary.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {

CeilOp::CeilOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : OneWayUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> CeilOp::clone() const {
  return std::make_unique<CeilOp>(*this);
}

std::vector<std::tuple<OperatorIdentifier, float>>
CeilOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::CeilInplace, 10}};
}

std::unique_ptr<Op>
CeilOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::CeilInplace) {
    return std::make_unique<CeilInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

CeilInplaceOp::CeilInplaceOp(const CeilOp &ceil_op)
    : OneWayUnaryInPlaceOp(Onnx::CustomOperators::CeilInplace,
                           ceil_op.getSettings()) {}

std::unique_ptr<Op> CeilInplaceOp::clone() const {
  return std::make_unique<CeilInplaceOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition ceilOpDef({OpDefinition::Inputs({{"X", T}}),
                               OpDefinition::Outputs({{"Y", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<CeilOp>
    ceilOpCreator(OpDefinitions({{Onnx::Operators::Ceil_1, ceilOpDef},
                                 {Onnx::Operators::Ceil_6, ceilOpDef}}));

} // namespace
} // namespace popart
