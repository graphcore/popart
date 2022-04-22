// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/div.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operators.hpp"

namespace popart {
struct OperatorIdentifier;

DivOp::DivOp(const OperatorIdentifier &_opid, const Op::Settings &_settings)
    : ElementWiseNpBroadcastableBinaryWithGradOp(_opid, _settings) {
  // TODO : Use the attributes in Div-6
}

std::unique_ptr<Op> DivOp::clone() const {
  return std::make_unique<DivOp>(*this);
}

DivArg0GradOp::DivArg0GradOp(const Op &op,
                             const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg0GradOp(Onnx::GradOperators::DivArg0Grad,
                                  _reduction_axes,
                                  op.inInfo(DivOp::getArg0InIndex()),
                                  op.getSettings()) {}

std::unique_ptr<Op> DivArg0GradOp::clone() const {
  return std::make_unique<DivArg0GradOp>(*this);
}

DivArg1GradOp::DivArg1GradOp(const Op &op,
                             const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg1GradOp(Onnx::GradOperators::DivArg1Grad,
                                  _reduction_axes,
                                  op.inInfo(DivOp::getArg1InIndex()),
                                  op.getSettings()) {}

std::unique_ptr<Op> DivArg1GradOp::clone() const {
  return std::make_unique<DivArg1GradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition divOpDef({OpDefinition::Inputs({
                                  {"A", T},
                                  {"B", T},
                              }),
                              OpDefinition::Outputs({{"C", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<DivOp> divOpCreator(OpDefinitions(
    {{Onnx::Operators::Div_6, divOpDef}, {Onnx::Operators::Div_7, divOpDef}}));
} // namespace

} // namespace popart
