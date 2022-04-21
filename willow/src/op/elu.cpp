// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/elu.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

EluOp::EluOp(const OperatorIdentifier &opid_,
             float alpha,
             const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid_, opSettings), alpha_(alpha) {}

std::unique_ptr<Op> EluOp::clone() const {
  return std::make_unique<EluOp>(*this);
}

std::vector<std::unique_ptr<Op>> EluOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<EluGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
EluOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::EluInplace, 10}};
}

std::unique_ptr<Op>
EluOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::EluInplace) {
    return std::make_unique<EluInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

void EluOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha_);
}

EluInplaceOp::EluInplaceOp(const EluOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::EluInplace,
                                op.getSettings()),
      alpha_(op.alpha()) {}

std::unique_ptr<Op> EluInplaceOp::clone() const {
  return std::make_unique<EluInplaceOp>(*this);
}

void EluInplaceOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha_);
}

EluGradOp::EluGradOp(const EluOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::EluGrad, fwdop),
      alpha_(fwdop.alpha()) {}

std::unique_ptr<Op> EluGradOp::clone() const {
  return std::make_unique<EluGradOp>(*this);
}

void EluGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("alpha", alpha_);
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

static OpDefinition eluOpDef({OpDefinition::Inputs({{"input", T}}),
                              OpDefinition::Outputs({{"output", T}}),
                              OpDefinition::Attributes({{"alpha", {"*"}}})});

static OpCreator<EluOp> eluOpCreator(
    OpDefinitions({
        {Onnx::Operators::Elu_1, eluOpDef},
        {Onnx::Operators::Elu_6, eluOpDef},
    }),
    [](const OpCreatorInfo &info) {
      float alpha =
          info.attributes.getAttribute<Attributes::Float>("alpha", 1.0f);

      return std::unique_ptr<Op>(new EluOp(info.opid, alpha, info.settings));
    },
    true);

} // namespace
} // namespace popart
