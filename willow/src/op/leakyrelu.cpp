// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/leakyrelu.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {

namespace {
constexpr const char *const ALPHA_ATTRIBUTE = "alpha";

// default alpha is 10**(-2) from
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu
const float ALPHA_DEFAULT = 1e-2f;
} // namespace

LeakyReluOp::LeakyReluOp(const OperatorIdentifier &_opid,
                         float _alpha,
                         const Op::Settings &_settings)
    : ElementWiseUnaryOp(_opid, _settings), LeakyReluOpBaseAttributes(_alpha) {}

std::unique_ptr<Op> LeakyReluOp::clone() const {
  return std::make_unique<LeakyReluOp>(*this);
}

std::vector<std::unique_ptr<popart::Op>> LeakyReluOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<LeakyReluGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
LeakyReluOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::LeakyReluInplace, 10}};
}

void LeakyReluOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute(ALPHA_ATTRIBUTE, getAlpha());
}

void LeakyReluOp::appendOutlineAttributes(popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute(ALPHA_ATTRIBUTE, getAlpha());
}

std::unique_ptr<Op>
LeakyReluOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::LeakyReluInplace) {
    return std::make_unique<LeakyReluInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

LeakyReluInplaceOp::LeakyReluInplaceOp(const LeakyReluOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::LeakyReluInplace,
                                op.getSettings()),
      LeakyReluOpBaseAttributes(op.getAlpha()) {}

std::unique_ptr<Op> LeakyReluInplaceOp::clone() const {
  return std::make_unique<LeakyReluInplaceOp>(*this);
}

void LeakyReluInplaceOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute(ALPHA_ATTRIBUTE, getAlpha());
}

void LeakyReluInplaceOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute(ALPHA_ATTRIBUTE, getAlpha());
}

LeakyReluGradOp::LeakyReluGradOp(const LeakyReluOp &fwdop)
    : Op(Onnx::GradOperators::LeakyReluGrad, fwdop.getSettings()),
      LeakyReluOpBaseAttributes(fwdop.getAlpha()) {}

void LeakyReluGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradLeakyReluInIndex());
}

const std::vector<GradInOutMapper> &LeakyReluGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradLeakyReluInIndex(),
       LeakyReluOp::getOutIndex(),
       GradOpInType::GradOut},
      {getLeakyReluInIndex(), LeakyReluOp::getOutIndex(), GradOpInType::Out}};
  return inInfo;
}

const std::map<int, int> &LeakyReluGradOp::gradOutToNonGradIn() const {
  // the grad-op's output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), LeakyReluOp::getInIndex()}};
  return outInfo;
}

void LeakyReluGradOp::appendAttributes(popart::OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute(ALPHA_ATTRIBUTE, getAlpha());
}

void LeakyReluGradOp::appendOutlineAttributes(
    popart::OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute(ALPHA_ATTRIBUTE, getAlpha());
}

std::unique_ptr<Op> LeakyReluGradOp::clone() const {
  return std::make_unique<LeakyReluGradOp>(*this);
}

namespace {
static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    leakyReluOpDef({OpDefinition::Inputs({{"input", T}}),
                    OpDefinition::Outputs({{"output", T}}),
                    OpDefinition::Attributes({{ALPHA_ATTRIBUTE, {"*"}}})});

static OpCreator<LeakyReluOp> leakyReluOpCreator(
    popart::OpDefinitions({{Onnx::Operators::LeakyRelu_1, leakyReluOpDef},
                           {Onnx::Operators::LeakyRelu_6, leakyReluOpDef}}),
    [](const OpCreatorInfo &info) {
      float alpha = info.attributes.getAttribute<popart::Attributes::Float>(
          ALPHA_ATTRIBUTE, ALPHA_DEFAULT);

      return std::make_unique<LeakyReluOp>(info.opid, alpha, info.settings);
    },
    true);
} // namespace

} // namespace popart
