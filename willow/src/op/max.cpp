// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/max.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

MaxOp::MaxOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : VariadicOp(_opid, settings_) {}

std::unique_ptr<Op> MaxOp::clone() const {
  return std::make_unique<MaxOp>(*this);
}

std::unique_ptr<Op> MaxOp::getIthGrad(int i) const {
  return std::make_unique<MaxArgGradOp>(*this, i);
}

MaxArgGradOp::MaxArgGradOp(const MaxOp &op_, InIndex inputIndex)
    : NonLinearVariadicGradOp(Onnx::GradOperators::MaxArgGrad,
                              op_,
                              inputIndex) {}

std::unique_ptr<Op> MaxArgGradOp::clone() const {
  return std::make_unique<MaxArgGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition maxOpDef({OpDefinition::Inputs({{"data", T}}),
                              OpDefinition::Outputs({{"max", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<MaxOp> maxOpCreator(OpDefinitions(
    {{Onnx::Operators::Max_6, maxOpDef}, {Onnx::Operators::Max_8, maxOpDef}}));
} // namespace

} // namespace popart
