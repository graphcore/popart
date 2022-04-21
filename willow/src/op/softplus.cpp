// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <vector>
#include <popart/op/softplus.hpp>
#include <popart/opmanager.hpp>

namespace popart {

SoftPlusOp::SoftPlusOp(const OperatorIdentifier &opid_,
                       const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid_, opSettings) {}

std::unique_ptr<Op> SoftPlusOp::clone() const {
  return std::make_unique<SoftPlusOp>(*this);
}

std::vector<std::unique_ptr<Op>> SoftPlusOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<SoftPlusGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
SoftPlusOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::SoftPlusInplace, 10}};
}

std::unique_ptr<Op>
SoftPlusOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SoftPlusInplace) {
    return std::make_unique<SoftPlusInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

SoftPlusInplaceOp::SoftPlusInplaceOp(const SoftPlusOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::SoftPlusInplace,
                                op.getSettings()) {}

std::unique_ptr<Op> SoftPlusInplaceOp::clone() const {
  return std::make_unique<SoftPlusInplaceOp>(*this);
}

SoftPlusGradOp::SoftPlusGradOp(const SoftPlusOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::SoftPlusGrad,
                                      fwdop) {}

std::unique_ptr<Op> SoftPlusGradOp::clone() const {
  return std::make_unique<SoftPlusGradOp>(*this);
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

static OpDefinition softplusOpDef({OpDefinition::Inputs({{"input", T}}),
                                   OpDefinition::Outputs({{"output", T}}),
                                   OpDefinition::Attributes({{}})});

static OpCreator<SoftPlusOp> softplusOpCreator(
    OpDefinitions({{Onnx::Operators::Softplus_1, softplusOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new SoftPlusOp(info.opid, info.settings));
    },
    true);

} // namespace
} // namespace popart
