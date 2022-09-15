// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/softsign.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {

SoftSignOp::SoftSignOp(const OperatorIdentifier &opid_,
                       const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid_, opSettings) {}

std::unique_ptr<Op> SoftSignOp::clone() const {
  return std::make_unique<SoftSignOp>(*this);
}

std::vector<std::unique_ptr<Op>> SoftSignOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<SoftSignGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
SoftSignOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::SoftSignInplace, 10}};
}

std::unique_ptr<Op>
SoftSignOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SoftSignInplace) {
    return std::make_unique<SoftSignInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

SoftSignInplaceOp::SoftSignInplaceOp(const SoftSignOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::SoftSignInplace,
                                op.getSettings()) {}

std::unique_ptr<Op> SoftSignInplaceOp::clone() const {
  return std::make_unique<SoftSignInplaceOp>(*this);
}

SoftSignGradOp::SoftSignGradOp(const SoftSignOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::SoftSignGrad,
                                      fwdop) {}

std::unique_ptr<Op> SoftSignGradOp::clone() const {
  return std::make_unique<SoftSignGradOp>(*this);
}

namespace {
static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition softsignOpDef({OpDefinition::Inputs({{"input", T}}),
                                   OpDefinition::Outputs({{"output", T}}),
                                   OpDefinition::Attributes({{}})});

static OpCreator<SoftSignOp> softsignOpCreator(
    OpDefinitions({{Onnx::Operators::Softsign_1, softsignOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new SoftSignOp(info.opid, info.settings));
    },
    true);

} // namespace
} // namespace popart
