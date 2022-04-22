// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/relu.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
ReluOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::ReluInplace, 10}};
}

std::unique_ptr<Op>
ReluOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ReluInplace) {
    return std::make_unique<ReluInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

ReluInplaceOp::ReluInplaceOp(const ReluOp &relu_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::ReluInplace,
                                relu_op.getSettings()) {}

ReluInplaceOp::ReluInplaceOp(const Settings &settings)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::ReluInplace, settings) {}

std::unique_ptr<Op> ReluInplaceOp::clone() const {
  return std::make_unique<ReluInplaceOp>(*this);
}

std::unique_ptr<Op> ReluOp::clone() const {
  return std::make_unique<ReluOp>(*this);
}

ReluOp::ReluOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::vector<std::unique_ptr<Op>> ReluOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<ReluGradOp>(*this));
  return upops;
}

void ReluGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradReludInIndex());
}

std::unique_ptr<Op> ReluGradOp::clone() const {
  return std::make_unique<ReluGradOp>(*this);
}

ReluGradOp::ReluGradOp(const ReluOp &op_)
    : Op(Onnx::GradOperators::ReluGrad, op_.getSettings()) {}

const std::vector<GradInOutMapper> &ReluGradOp::gradInputInfo() const {
  // input at index getGradReludIn() (=0) : gradient of output of relu
  // input at index getReludIn() (=1)     : output of relu
  // can we do better sometimes with in-placing?
  // The 0's below : As there is only 1 output of Relu, it
  // is output at index 0.
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradReludInIndex(), ReluOp::getOutIndex(), GradOpInType::GradOut},
      {getReludInIndex(), ReluOp::getOutIndex(), GradOpInType::Out}};
  return inInfo;
}

const std::map<int, int> &ReluGradOp::gradOutToNonGradIn() const {
  // the grad-op's output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ReluOp::getInIndex()}};
  return outInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition reluOpDef({OpDefinition::Inputs({{"X", T}}),
                               OpDefinition::Outputs({{"Y", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<ReluOp> reluOpCreator(OpDefinitions({
    {Onnx::Operators::Relu_6, reluOpDef},
}));
} // namespace

} // namespace popart
