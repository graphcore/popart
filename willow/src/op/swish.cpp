// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/swish.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

SwishOp::SwishOp(const OperatorIdentifier &opid, const Op::Settings &settings)
    : ElementWiseUnaryOp(opid, settings) {}

std::unique_ptr<Op> SwishOp::clone() const {
  return std::make_unique<SwishOp>(*this);
}

std::vector<std::unique_ptr<Op>> SwishOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<SwishGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
SwishOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::SwishInplace, 10}};
}

std::unique_ptr<Op>
SwishOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SwishInplace) {
    return std::make_unique<SwishInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

SwishInplaceOp::SwishInplaceOp(const SwishOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::SwishInplace,
                                op.getSettings()) {}

std::unique_ptr<Op> SwishInplaceOp::clone() const {
  return std::make_unique<SwishInplaceOp>(*this);
}

SwishGradOp::SwishGradOp(const SwishOp &fwdOp)
    : ElementWiseNonLinearUnaryGradOp(Onnx::CustomGradOperators::SwishGrad,
                                      fwdOp) {}

std::unique_ptr<Op> SwishGradOp::clone() const {
  return std::make_unique<SwishGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition swishOpDef({OpDefinition::Inputs({{"input", T}}),
                                OpDefinition::Outputs({{"output", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<SwishOp> swishOpCreator(
    OpDefinitions({{Onnx::AiGraphcore::OpSet1::Swish, swishOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new SwishOp(info.opid, info.settings));
    },
    true);
} // namespace

} // namespace popart
