// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/nearbyint.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/onewayunary.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {

std::unique_ptr<Op> NearbyIntOp::clone() const {
  return std::make_unique<NearbyIntOp>(*this);
}

std::unique_ptr<Op>
NearbyIntOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::NearbyIntInplace) {
    return std::make_unique<NearbyIntInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

std::vector<std::tuple<OperatorIdentifier, float>>
NearbyIntOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::NearbyIntInplace, 10}};
}

NearbyIntInplaceOp::NearbyIntInplaceOp(const NearbyIntOp &NearbyInt_op)
    : OneWayUnaryInPlaceOp(Onnx::CustomOperators::NearbyIntInplace,
                           NearbyInt_op.getSettings()) {}

std::unique_ptr<Op> NearbyIntInplaceOp::clone() const {
  return std::make_unique<NearbyIntInplaceOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition nearbyintOpDef({OpDefinition::Inputs({{"X", T}}),
                                    OpDefinition::Outputs({{"Y", T}}),
                                    OpDefinition::Attributes({})});

static OpCreator<NearbyIntOp> NearbyIntOpCreator(
    OpDefinitions({{Onnx::CustomOperators::NearbyInt, nearbyintOpDef}}));

} // namespace
} // namespace popart
