// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <popart/op/sign.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/onewayunary.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"

namespace popart {
class Ir;

SignOp::SignOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : OneWayUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> SignOp::clone() const {
  return std::make_unique<SignOp>(*this);
}

OperatorIdentifier SignOp::getOpId(const Ir &) {
  return Onnx::Operators::Sign_9;
}

SignInplaceOp::SignInplaceOp(const SignOp &sign_op)
    : OneWayUnaryInPlaceOp(Onnx::CustomOperators::SignInplace,
                           sign_op.getSettings()) {}

std::unique_ptr<Op> SignInplaceOp::clone() const {
  return std::make_unique<SignInplaceOp>(*this);
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

static OpDefinition signOpDef({OpDefinition::Inputs({{"input", T}}),
                               OpDefinition::Outputs({{"output", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<SignOp> signOpCreator(OpDefinitions({
    {Onnx::Operators::Sign_9, signOpDef},
}));
} // namespace

} // namespace popart
