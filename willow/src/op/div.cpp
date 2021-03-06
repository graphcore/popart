// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/div.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

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

DivArg1GradOp::DivArg1GradOp(const Op &op,
                             const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg1GradOp(Onnx::GradOperators::DivArg1Grad,
                                  _reduction_axes,
                                  op.inInfo(DivOp::getArg1InIndex()),
                                  op.getSettings()) {}

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
