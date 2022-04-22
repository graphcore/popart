// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/sqrt.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

SqrtOp::SqrtOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> SqrtOp::clone() const {
  return std::make_unique<SqrtOp>(*this);
}

std::vector<std::unique_ptr<Op>> SqrtOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SqrtGradOp>(*this));
  return upops;
}

SqrtGradOp::SqrtGradOp(const SqrtOp &fwdOp)
    : Op(Onnx::GradOperators::SqrtGrad, fwdOp.getSettings()) {}

std::unique_ptr<Op> SqrtGradOp::clone() const {
  return std::make_unique<SqrtGradOp>(*this);
}

void SqrtGradOp::setup() { outInfo(getOutIndex()) = inInfo(getGradInIndex()); }

const std::vector<GradInOutMapper> &SqrtGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), 0, GradOpInType::GradOut},
      {getFwdOutInIndex(), SqrtOp::getOutIndex(), GradOpInType::Out}};

  return inInfo;
}

const std::map<int, int> &SqrtGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SqrtOp::getInIndex()}};

  return outInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition sqrtOpDef({OpDefinition::Inputs({{"X", T}}),
                               OpDefinition::Outputs({{"Y", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<SqrtOp> sqrtOpCreator(OpDefinitions({
    {Onnx::Operators::Sqrt_6, sqrtOpDef},
}));
} // namespace

} // namespace popart
