// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/hardsigmoid.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {

HardSigmoidOp::HardSigmoidOp(const OperatorIdentifier &opid_,
                             float _alpha,
                             float _beta,
                             const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid_, opSettings), alpha(_alpha), beta(_beta) {}

std::unique_ptr<Op> HardSigmoidOp::clone() const {
  return std::make_unique<HardSigmoidOp>(*this);
}

std::vector<std::unique_ptr<Op>> HardSigmoidOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<HardSigmoidGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
HardSigmoidOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::HardSigmoidInplace, 10}};
}

std::unique_ptr<Op>
HardSigmoidOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::HardSigmoidInplace) {
    return std::make_unique<HardSigmoidInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

void HardSigmoidOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha);
  os.appendAttribute("beta", beta);
}

HardSigmoidInplaceOp::HardSigmoidInplaceOp(const HardSigmoidOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::HardSigmoidInplace,
                                op.getSettings()),
      alpha(op.getAlpha()), beta(op.getBeta()) {}

std::unique_ptr<Op> HardSigmoidInplaceOp::clone() const {
  return std::make_unique<HardSigmoidInplaceOp>(*this);
}

void HardSigmoidInplaceOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha);
  os.appendAttribute("beta", beta);
}

HardSigmoidGradOp::HardSigmoidGradOp(const HardSigmoidOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::HardSigmoidGrad,
                                      fwdop),
      alpha(fwdop.getAlpha()), beta(fwdop.getBeta()) {}

std::unique_ptr<Op> HardSigmoidGradOp::clone() const {
  return std::make_unique<HardSigmoidGradOp>(*this);
}

void HardSigmoidGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha);
  os.appendAttribute("beta", beta);
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

static OpDefinition hardsigmoidOpDef(
    {OpDefinition::Inputs({{"input", T}}),
     OpDefinition::Outputs({{"output", T}}),
     OpDefinition::Attributes({{"alpha", {"*"}}, {"beta", {"*"}}})});

static OpCreator<HardSigmoidOp> hardsigmoidOpCreator(
    OpDefinitions({
        {Onnx::Operators::HardSigmoid_1, hardsigmoidOpDef},
        {Onnx::Operators::HardSigmoid_6, hardsigmoidOpDef},
    }),
    [](const OpCreatorInfo &info) {
      float alpha =
          info.attributes.getAttribute<Attributes::Float>("alpha", 0.2f);
      float beta =
          info.attributes.getAttribute<Attributes::Float>("beta", 0.5f);

      return std::unique_ptr<Op>(
          new HardSigmoidOp(info.opid, alpha, beta, info.settings));
    },
    true);

} // namespace
} // namespace popart
