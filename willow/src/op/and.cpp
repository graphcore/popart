// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <vector>
#include <popart/op/and.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

AndOp::AndOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> AndOp::clone() const {
  return std::make_unique<AndOp>(*this);
}

std::vector<std::unique_ptr<Op>> AndOp::getGradOps() {
  throw error("PopART does not have a valid grad op corresponding to AndOp");
}

namespace {

static OpDefinition::DataTypes T  = {DataType::BOOL};
static OpDefinition::DataTypes T1 = {DataType::BOOL};

static OpDefinition andOpDef({OpDefinition::Inputs({{"A", T}, {"B", T}}),
                              OpDefinition::Outputs({{"C", T1}}),
                              OpDefinition::Attributes({})});

static OpCreator<AndOp> AndOpCreator(OpDefinitions(
    {{Onnx::Operators::And_1, andOpDef}, {Onnx::Operators::And_7, andOpDef}}));

} // namespace

} // namespace popart
