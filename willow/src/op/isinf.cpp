// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <popart/op/isinf.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {
class Ir;

IsInf::IsInf(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryBooleanOp(_opid, settings_) {}

std::unique_ptr<Op> IsInf::clone() const {
  return std::make_unique<IsInf>(*this);
}

OperatorIdentifier IsInf::getOpId(const Ir &) {
  return Onnx::Operators::IsInf_10;
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT};
static OpDefinition::DataTypes T2 = {DataType::BOOL};

static OpDefinition
    isInfOpDef({OpDefinition::Inputs({{"x", T1}}),
                OpDefinition::Outputs({{"y", T2}}),
                OpDefinition::Attributes({
                    // Dont support the optional attributes. T13477
                    //{"detect_negative", {"?"}},
                    //{"detect_positive", {"?"}},

                })});

static OpCreator<IsInf>
    IsInfCreator(OpDefinitions({{Onnx::Operators::IsInf_10, isInfOpDef}}));
} // namespace

} // namespace popart
