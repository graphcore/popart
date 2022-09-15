// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/gelu.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

GeluOp::GeluOp(const OperatorIdentifier &opid_, const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid_, opSettings) {}

std::unique_ptr<Op> GeluOp::clone() const {
  return std::make_unique<GeluOp>(*this);
}

std::vector<std::unique_ptr<Op>> GeluOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<GeluGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
GeluOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::GeluInplace, 10}};
}

std::unique_ptr<Op>
GeluOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::GeluInplace) {
    return std::make_unique<GeluInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

GeluInplaceOp::GeluInplaceOp(const GeluOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::GeluInplace,
                                op.getSettings()) {}

GeluInplaceOp::GeluInplaceOp(const Op::Settings &opSettings)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::GeluInplace,
                                opSettings) {}

std::unique_ptr<Op> GeluInplaceOp::clone() const {
  return std::make_unique<GeluInplaceOp>(*this);
}

GeluGradOp::GeluGradOp(const GeluOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::GeluGrad, fwdop) {}

std::unique_ptr<Op> GeluGradOp::clone() const {
  return std::make_unique<GeluGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition geluOpDef({OpDefinition::Inputs({{"input", T}}),
                               OpDefinition::Outputs({{"output", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<GeluOp> geluOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Gelu_1, geluOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new GeluOp(info.opid, info.settings));
    },
    true);

} // namespace
} // namespace popart
