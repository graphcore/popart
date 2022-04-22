// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <popart/op/min.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/variadic.hpp"
#include "popart/operators.hpp"

namespace popart {
struct OperatorIdentifier;

MinOp::MinOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : VariadicOp(_opid, settings_) {}

std::unique_ptr<Op> MinOp::clone() const {
  return std::make_unique<MinOp>(*this);
}

std::unique_ptr<Op> MinOp::getIthGrad(int i) const {
  return std::make_unique<MinArgGradOp>(*this, i);
}

MinArgGradOp::MinArgGradOp(const MinOp &op_, InIndex inputIndex)
    : NonLinearVariadicGradOp(Onnx::GradOperators::MinArgGrad,
                              op_,
                              inputIndex) {}

std::unique_ptr<Op> MinArgGradOp::clone() const {
  return std::make_unique<MinArgGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition minOpDef({OpDefinition::Inputs({{"data_0", T}}),
                              OpDefinition::Outputs({{"min", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<MinOp> minOpCreator(OpDefinitions(
    {{Onnx::Operators::Min_6, minOpDef}, {Onnx::Operators::Min_8, minOpDef}}));
} // namespace

} // namespace popart
