// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/round.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/onewayunary.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {

RoundOp::RoundOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : OneWayUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> RoundOp::clone() const {
  return std::make_unique<RoundOp>(*this);
}

std::unique_ptr<Op>
RoundOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::RoundInplace) {
    return std::make_unique<RoundInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

std::vector<std::tuple<OperatorIdentifier, float>>
RoundOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::RoundInplace, 10}};
}

RoundInplaceOp::RoundInplaceOp(const RoundOp &Round_op)
    : OneWayUnaryInPlaceOp(Onnx::CustomOperators::RoundInplace,
                           Round_op.getSettings()) {}

std::unique_ptr<Op> RoundInplaceOp::clone() const {
  return std::make_unique<RoundInplaceOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition roundOpDef({OpDefinition::Inputs({{"X", T}}),
                                OpDefinition::Outputs({{"Y", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<RoundOp> RoundOpCreator(
    OpDefinitions({{Onnx::Operators::Round_11, roundOpDef},
                   {Onnx::CustomOperators::Round_1, roundOpDef}}));

} // namespace
} // namespace popart
