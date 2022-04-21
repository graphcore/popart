// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/selu.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

SeluOp::SeluOp(const OperatorIdentifier &opid_,
               float _alpha,
               float _gamma,
               const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid_, opSettings), alpha(_alpha), gamma(_gamma) {}

std::unique_ptr<Op> SeluOp::clone() const {
  return std::make_unique<SeluOp>(*this);
}

std::vector<std::unique_ptr<Op>> SeluOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<SeluGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
SeluOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::SeluInplace, 10}};
}

std::unique_ptr<Op>
SeluOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SeluInplace) {
    return std::make_unique<SeluInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

void SeluOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha);
  os.appendAttribute("gamma", gamma);
}

SeluInplaceOp::SeluInplaceOp(const SeluOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::SeluInplace,
                                op.getSettings()),
      alpha(op.getAlpha()), gamma(op.getGamma()) {}

std::unique_ptr<Op> SeluInplaceOp::clone() const {
  return std::make_unique<SeluInplaceOp>(*this);
}

void SeluInplaceOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha);
  os.appendAttribute("gamma", gamma);
}

SeluGradOp::SeluGradOp(const SeluOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::SeluGrad, fwdop),
      alpha(fwdop.getAlpha()), gamma(fwdop.getGamma()) {}

std::unique_ptr<Op> SeluGradOp::clone() const {
  return std::make_unique<SeluGradOp>(*this);
}

void SeluGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha);
  os.appendAttribute("gamma", gamma);
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

static OpDefinition
    seluOpDef({OpDefinition::Inputs({{"input", T}}),
               OpDefinition::Outputs({{"output", T}}),
               OpDefinition::Attributes({{"alpha", {"*"}}, {"beta", {"*"}}})});

static OpCreator<SeluOp> seluOpCreator(
    OpDefinitions({
        {Onnx::Operators::Selu_1, seluOpDef},
        {Onnx::Operators::Selu_6, seluOpDef},
    }),
    [](const OpCreatorInfo &info) {
      float alpha = info.attributes.getAttribute<Attributes::Float>(
          "alpha", 1.67326319217681884765625f);
      float gamma = info.attributes.getAttribute<Attributes::Float>(
          "gamma", 1.05070102214813232421875f);

      return std::unique_ptr<Op>(
          new SeluOp(info.opid, alpha, gamma, info.settings));
    },
    true);

} // namespace
} // namespace popart
