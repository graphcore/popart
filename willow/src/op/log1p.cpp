// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/log1p.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
Log1pOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::Log1pInplace, 10}};
}

std::unique_ptr<Op>
Log1pOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::Log1pInplace) {
    return std::make_unique<Log1pInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

Log1pInplaceOp::Log1pInplaceOp(const Log1pOp &exp_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::Log1pInplace,
                                exp_op.getSettings()) {}

std::unique_ptr<Op> Log1pInplaceOp::clone() const {
  return std::make_unique<Log1pInplaceOp>(*this);
}

Log1pOp::Log1pOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> Log1pOp::clone() const {
  return std::make_unique<Log1pOp>(*this);
}

std::vector<std::unique_ptr<Op>> Log1pOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<Log1pGradOp>(*this));
  return result;
}

Log1pGradOp::Log1pGradOp(const Log1pOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::Log1pGrad, fwdop) {}

std::unique_ptr<Op> Log1pGradOp::clone() const {
  return std::make_unique<Log1pGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition log1pOpDef({OpDefinition::Inputs({{"input", T}}),
                                OpDefinition::Outputs({{"output", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<Log1pOp> log1pOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Log1p_1, log1pOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new Log1pOp(info.opid, info.settings));
    },
    true);

} // namespace

} // namespace popart
