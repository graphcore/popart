// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <popart/op/isnan.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {
class Ir;

IsNaN::IsNaN(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryBooleanOp(_opid, settings_) {}

std::unique_ptr<Op> IsNaN::clone() const {
  return std::make_unique<IsNaN>(*this);
}

OperatorIdentifier IsNaN::getOpId(const Ir &) {
  return Onnx::Operators::IsNaN_9;
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT, DataType::FLOAT16};
static OpDefinition::DataTypes T2 = {DataType::BOOL};

static OpDefinition isNanOpDef({OpDefinition::Inputs({{"x", T1}}),
                                OpDefinition::Outputs({{"y", T2}}),
                                OpDefinition::Attributes({})});

static OpCreator<IsNaN>
    IsNaNCreator(OpDefinitions({{Onnx::Operators::IsNaN_9, isNanOpDef}}));
} // namespace

} // namespace popart
