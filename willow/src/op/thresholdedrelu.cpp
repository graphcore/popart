// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/thresholdedrelu.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

ThresholdedReluOp::ThresholdedReluOp(const OperatorIdentifier &opid_,
                                     float _alpha,
                                     const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid_, opSettings), alpha(_alpha) {}

std::unique_ptr<Op> ThresholdedReluOp::clone() const {
  return std::make_unique<ThresholdedReluOp>(*this);
}

std::vector<std::unique_ptr<Op>> ThresholdedReluOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<ThresholdedReluGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
ThresholdedReluOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::ThresholdedReluInplace, 10}};
}

std::unique_ptr<Op> ThresholdedReluOp::getInplaceVariant(
    const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ThresholdedReluInplace) {
    return std::make_unique<ThresholdedReluInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

void ThresholdedReluOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha);
}

ThresholdedReluInplaceOp::ThresholdedReluInplaceOp(const ThresholdedReluOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::ThresholdedReluInplace,
                                op.getSettings()),
      alpha(op.getAlpha()) {}

std::unique_ptr<Op> ThresholdedReluInplaceOp::clone() const {
  return std::make_unique<ThresholdedReluInplaceOp>(*this);
}

void ThresholdedReluInplaceOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha);
}

ThresholdedReluGradOp::ThresholdedReluGradOp(const ThresholdedReluOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::ThresholdedReluGrad,
                                      fwdop),
      alpha(fwdop.getAlpha()) {}

std::unique_ptr<Op> ThresholdedReluGradOp::clone() const {
  return std::make_unique<ThresholdedReluGradOp>(*this);
}

void ThresholdedReluGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha);
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
    thresholdedreluOpDef({OpDefinition::Inputs({{"input", T}}),
                          OpDefinition::Outputs({{"output", T}}),
                          OpDefinition::Attributes({{"alpha", {"*"}}})});

static OpCreator<ThresholdedReluOp> thresholdedreluOpCreator(
    OpDefinitions({
        {Onnx::Operators::ThresholdedRelu_10, thresholdedreluOpDef},
    }),
    [](const OpCreatorInfo &info) {
      float alpha =
          info.attributes.getAttribute<Attributes::Float>("alpha", 1.0f);

      return std::unique_ptr<Op>(
          new ThresholdedReluOp(info.opid, alpha, info.settings));
    },
    true);

} // namespace
} // namespace popart
