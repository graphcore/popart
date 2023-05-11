// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/op/geluerf.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/opmanager.hpp"

namespace popart {

GeluErfOp::GeluErfOp(const OperatorIdentifier &opid_,
                     const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid_, opSettings) {}

std::unique_ptr<Op> GeluErfOp::clone() const {
  return std::make_unique<GeluErfOp>(*this);
}

std::vector<std::unique_ptr<Op>> GeluErfOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<GeluErfGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
GeluErfOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::GeluErfInplace, 10}};
}

std::unique_ptr<Op>
GeluErfOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::GeluErfInplace) {
    return std::make_unique<GeluErfInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

GeluErfInplaceOp::GeluErfInplaceOp(const GeluErfOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::GeluErfInplace,
                                op.getSettings()) {}

GeluErfInplaceOp::GeluErfInplaceOp(const Op::Settings &opSettings)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::GeluErfInplace,
                                opSettings) {}

std::unique_ptr<Op> GeluErfInplaceOp::clone() const {
  return std::make_unique<GeluErfInplaceOp>(*this);
}

GeluErfGradOp::GeluErfGradOp(const GeluErfOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::GeluErfGrad, fwdop) {
}

std::unique_ptr<Op> GeluErfGradOp::clone() const {
  return std::make_unique<GeluErfGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition geluErfOpDef({OpDefinition::Inputs({{"input", T}}),
                                  OpDefinition::Outputs({{"output", T}}),
                                  OpDefinition::Attributes({})});

static OpCreator<GeluErfOp> geluErfOpCreator(
    OpDefinitions({{Onnx::CustomOperators::GeluErf_1, geluErfOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new GeluErfOp(info.opid, info.settings));
    },
    true);

} // namespace
} // namespace popart
