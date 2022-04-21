// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/subtract.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

SubtractOp::SubtractOp(const OperatorIdentifier &_opid,
                       const Op::Settings &_settings)
    : ElementWiseNpBroadcastableBinaryWithGradOp(_opid, _settings) {
  // TODO : Do not broadcast in version 6
}

std::unique_ptr<Op> SubtractOp::clone() const {
  return std::make_unique<SubtractOp>(*this);
}

SubtractArg0GradOp::SubtractArg0GradOp(
    const Op &op,
    const std::vector<int64_t> &_reduction_axes)
    : ReduceSumOp(Onnx::GradOperators::SubArg0Grad,
                  _reduction_axes,
                  false,
                  op.getSettings()),
      forward_op_arg_info(op.inInfo(SubtractOp::getArg0InIndex())) {}

std::unique_ptr<Op> SubtractArg0GradOp::clone() const {
  return std::make_unique<SubtractArg0GradOp>(*this);
}

const std::map<int, int> &SubtractArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, SubtractOp::getArg0InIndex()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, SubtractOp::getOutIndex(), GradOpInType::GradOut}};

  return inInfo;
}

void SubtractArg0GradOp::setup() {
  outInfo(getOutIndex()) = forward_op_arg_info;
}

SubtractArg1GradOp::SubtractArg1GradOp(
    const Op &op,
    const std::vector<int64_t> &_reduction_axes)
    : ElementWiseBinaryArg1GradOp(Onnx::GradOperators::SubArg1Grad,
                                  _reduction_axes,
                                  op.inInfo(SubtractOp::getArg1InIndex()),
                                  op.getSettings()) {}

std::unique_ptr<Op> SubtractArg1GradOp::clone() const {
  return std::make_unique<SubtractArg1GradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition substractOpDef({OpDefinition::Inputs({
                                        {"A", T},
                                        {"B", T},
                                    }),
                                    OpDefinition::Outputs({{"C", T}}),
                                    OpDefinition::Attributes({})});

static OpCreator<SubtractOp> subtractOpCreator(
    OpDefinitions({{Onnx::Operators::Sub_6, substractOpDef},
                   {Onnx::Operators::Sub_7, substractOpDef}}));

} // namespace

} // namespace popart
