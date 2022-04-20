// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/shrink.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {

ShrinkOp::ShrinkOp(const OperatorIdentifier &opid_,
                   float lambd,
                   float bias,
                   const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid_, opSettings), lambd_(lambd), bias_(bias) {}

std::unique_ptr<Op> ShrinkOp::clone() const {
  return std::make_unique<ShrinkOp>(*this);
}

std::vector<std::unique_ptr<Op>> ShrinkOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<ShrinkGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
ShrinkOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::ShrinkInplace, 10}};
}

std::unique_ptr<Op>
ShrinkOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ShrinkInplace) {
    return std::make_unique<ShrinkInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

void ShrinkOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("lambd", lambd_);
  os.appendAttribute("bias", bias_);
}

ShrinkInplaceOp::ShrinkInplaceOp(const ShrinkOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::ShrinkInplace,
                                op.getSettings()),
      lambd_(op.lambd()), bias_(op.bias()) {}

std::unique_ptr<Op> ShrinkInplaceOp::clone() const {
  return std::make_unique<ShrinkInplaceOp>(*this);
}

void ShrinkInplaceOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("lambd", lambd_);
  os.appendAttribute("bias", bias_);
}

ShrinkGradOp::ShrinkGradOp(const ShrinkOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::ShrinkGrad, fwdop),
      lambd_(fwdop.lambd()), bias_(fwdop.bias()) {}

std::unique_ptr<Op> ShrinkGradOp::clone() const {
  return std::make_unique<ShrinkGradOp>(*this);
}

void ShrinkGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("lambd", lambd_);
  os.appendAttribute("bias", bias_);
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

static OpDefinition shrinkOpDef({OpDefinition::Inputs({{"input", T}}),
                                 OpDefinition::Outputs({{"output", T}}),
                                 OpDefinition::Attributes({{"bias", {"*"}},
                                                           {"lambd", {"*"}}})});

static OpCreator<ShrinkOp> shrinkOpCreator(
    OpDefinitions({
        {Onnx::Operators::Shrink_9, shrinkOpDef},
    }),
    [](const OpCreatorInfo &info) {
      float lambd =
          info.attributes.getAttribute<Attributes::Float>("lambd", 0.5f);
      float bias =
          info.attributes.getAttribute<Attributes::Float>("bias", 0.0f);

      return std::unique_ptr<Op>(
          new ShrinkOp(info.opid, lambd, bias, info.settings));
    },
    true);

} // namespace
} // namespace popart
