// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/op/bitwise.hpp"

#include <memory>
#include <string>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"

namespace popart {
struct OperatorIdentifier;

BitwiseNotOp::BitwiseNotOp(const OperatorIdentifier &_opid,
                           const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> BitwiseNotOp::clone() const {
  return std::make_unique<BitwiseNotOp>(*this);
}

std::vector<std::unique_ptr<Op>> BitwiseNotOp::getGradOps() {
  throw error(
      "PopART does not have a valid grad op corresponding to BitwiseNotOp");
}

BitwiseBinaryOp::BitwiseBinaryOp(const OperatorIdentifier &_opid,
                                 const Op::Settings &settings_)
    : ElementWiseBinaryOp(_opid, settings_) {}

std::unique_ptr<Op> BitwiseBinaryOp::clone() const {
  return std::make_unique<BitwiseBinaryOp>(*this);
}

std::vector<std::unique_ptr<Op>> BitwiseBinaryOp::getGradOps() {
  throw error(
      "PopART does not have a valid grad op corresponding to BitwiseBinaryOp");
}

namespace {
static OpDefinition::DataTypes BitwiseT = {DataType::INT32, DataType::UINT32};

static OpDefinition bitwiseNotOptDef({OpDefinition::Inputs({{"X", BitwiseT}}),
                                      OpDefinition::Outputs({{"Y", BitwiseT}}),
                                      OpDefinition::Attributes({})});
static OpDefinition bitwiseBinaryOptDef(
    {OpDefinition::Inputs({{"X", BitwiseT}, {"Y", BitwiseT}}),
     OpDefinition::Outputs({{"Z", BitwiseT}}),
     OpDefinition::Attributes({})});

static OpCreator<BitwiseNotOp> BitwiseNotOpCreator(
    OpDefinitions({{Onnx::AiGraphcore::OpSet1::BitwiseNot, bitwiseNotOptDef}}));
static OpCreator<BitwiseBinaryOp> BitwiseAndOpCreator(OpDefinitions(
    {{Onnx::AiGraphcore::OpSet1::BitwiseAnd, bitwiseBinaryOptDef}}));
static OpCreator<BitwiseBinaryOp> BitwiseOrOpCreator(OpDefinitions(
    {{Onnx::AiGraphcore::OpSet1::BitwiseOr, bitwiseBinaryOptDef}}));
static OpCreator<BitwiseBinaryOp> BitwiseXorOpCreator(OpDefinitions(
    {{Onnx::AiGraphcore::OpSet1::BitwiseXor, bitwiseBinaryOptDef}}));
static OpCreator<BitwiseBinaryOp> BitwiseXnorOpCreator(OpDefinitions(
    {{Onnx::AiGraphcore::OpSet1::BitwiseXnor, bitwiseBinaryOptDef}}));

} // namespace
} // namespace popart
