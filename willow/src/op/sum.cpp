// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <vector>
#include <popart/op/sum.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/variadic.hpp"
#include "popart/operators.hpp"

namespace popart {
struct OperatorIdentifier;

SumOp::SumOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : VariadicOp(_opid, settings_) {
  // TODO : Do not broadcast in version 6
}

std::unique_ptr<Op> SumOp::clone() const {
  return std::make_unique<SumOp>(*this);
}

std::unique_ptr<Op> SumOp::getIthGrad(int i) const {
  return std::make_unique<SumArgGradOp>(*this, i);
}

SumArgGradOp::SumArgGradOp(const SumOp &op_, InIndex inputIndex)
    : LinearVariadicGradOp(Onnx::GradOperators::SumArgGrad, op_, inputIndex) {

  gradInputInfoVec = {
      {getGradInIndex(), VariadicOp::getOutIndex(), GradOpInType::GradOut}};
}

std::unique_ptr<Op> SumArgGradOp::clone() const {
  return std::make_unique<SumArgGradOp>(*this);
}

const std::vector<GradInOutMapper> &SumArgGradOp::gradInputInfo() const {
  return gradInputInfoVec;
}

bool SumArgGradOp::canBeReplacedByIdentity() const {
  return inShape(0) == outShape(0);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition sumOpDef({OpDefinition::Inputs({
                                  {"data_0", T},
                              }),
                              OpDefinition::Outputs({{"sum", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<SumOp> sumOpCreator(OpDefinitions(
    {{Onnx::Operators::Sum_6, sumOpDef}, {Onnx::Operators::Sum_8, sumOpDef}}));
} // namespace

} // namespace popart
