// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/expm1.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
Expm1Op::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::Expm1Inplace, 10}};
}

std::unique_ptr<Op>
Expm1Op::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::Expm1Inplace) {
    return std::make_unique<Expm1InplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

Expm1InplaceOp::Expm1InplaceOp(const Expm1Op &exp_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::Expm1Inplace,
                                exp_op.getSettings()) {}

std::unique_ptr<Op> Expm1InplaceOp::clone() const {
  return std::make_unique<Expm1InplaceOp>(*this);
}

Expm1Op::Expm1Op(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> Expm1Op::clone() const {
  return std::make_unique<Expm1Op>(*this);
}

std::vector<std::unique_ptr<Op>> Expm1Op::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<Expm1GradOp>(*this));
  return upops;
}

Expm1GradOp::Expm1GradOp(const Expm1Op &fwdOp)
    : Op(Onnx::GradOperators::Expm1Grad, fwdOp.getSettings()) {}

std::unique_ptr<Op> Expm1GradOp::clone() const {
  return std::make_unique<Expm1GradOp>(*this);
}

const std::vector<GradInOutMapper> &Expm1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), Expm1Op::getOutIndex(), GradOpInType::GradOut},
      {getFwdOutInIndex(), Expm1Op::getOutIndex(), GradOpInType::Out}};

  return inInfo;
}

const std::map<int, int> &Expm1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), Expm1Op::getInIndex()}};

  return outInfo;
}

void Expm1GradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getFwdOutInIndex());
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition expm1OpDef({OpDefinition::Inputs({{"input", T}}),
                                OpDefinition::Outputs({{"output", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<Expm1Op> expm1OpCreator(
    OpDefinitions({{Onnx::CustomOperators::Expm1_1, expm1OpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new Expm1Op(info.opid, info.settings));
    },
    true);

} // namespace

} // namespace popart
