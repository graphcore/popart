// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/div.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

DivOp::DivOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseBinaryOp(_opid, settings_) {
  // TODO : Use the attributes in Div-6
}

std::unique_ptr<Op> DivOp::clone() const {
  return std::make_unique<DivOp>(*this);
}

std::vector<std::unique_ptr<Op>> DivOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_in_0   = inShape(getArg0InIndex());
  const auto &shape_in_1   = inShape(getArg1InIndex());
  const auto &shape_output = outShape(getOutIndex());

  upops.emplace_back(std::make_unique<DivArg0GradOp>(
      *this, npReductionAxis(shape_in_0, shape_output)));
  upops.emplace_back(std::make_unique<DivArg1GradOp>(
      *this, npReductionAxis(shape_in_1, shape_output)));
  return upops;
}

DivArgGradOp::DivArgGradOp(const OperatorIdentifier &_opid,
                           const std::vector<int64_t> &reduction_axes_,
                           const TensorInfo &forward_op_arg_info_,
                           const Op::Settings &settings_)
    : Op(_opid, settings_), forward_op_arg_info(forward_op_arg_info_),
      reduction_axes(reduction_axes_) {}

void DivArgGradOp::setup() { outInfo(0) = forward_op_arg_info; }

const std::vector<int64_t> &DivArgGradOp::getReductionAxes() const {
  return reduction_axes;
}

DivArg0GradOp::DivArg0GradOp(const DivOp &op,
                             const std::vector<int64_t> &reduction_axes_)
    : DivArgGradOp(Onnx::GradOperators::DivArg0Grad,
                   reduction_axes_,
                   op.inInfo(DivOp::getArg0InIndex()),
                   op.getSettings()) {}

std::unique_ptr<Op> DivArg0GradOp::clone() const {
  return std::make_unique<DivArg0GradOp>(*this);
}

const std::map<int, int> &DivArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DivOp::getArg0InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &DivArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, getOutIndex(), GradOpInType::GRADOUT},
      {1, DivOp::getArg1InIndex(), GradOpInType::IN}};
  return inInfo;
}

DivArg1GradOp::DivArg1GradOp(const DivOp &op,
                             const std::vector<int64_t> &reduction_axes_)
    : DivArgGradOp(Onnx::GradOperators::DivArg1Grad,
                   reduction_axes_,
                   op.inInfo(DivOp::getArg1InIndex()),
                   op.getSettings()) {}

std::unique_ptr<Op> DivArg1GradOp::clone() const {
  return std::make_unique<DivArg1GradOp>(*this);
}

const std::map<int, int> &DivArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DivOp::getArg1InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &DivArg1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, getOutIndex(), GradOpInType::GRADOUT},
      {1, DivOp::getArg0InIndex(), GradOpInType::IN},
      {2, DivOp::getArg1InIndex(), GradOpInType::IN}};
  return inInfo;
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
