// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/op/logsoftmax.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

LogSoftmaxOp::LogSoftmaxOp(const OperatorIdentifier &_opid,
                           int64_t axis_,
                           const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_), axis(axis_) {}

std::unique_ptr<Op> LogSoftmaxOp::clone() const {
  return std::make_unique<LogSoftmaxOp>(*this);
}

std::vector<std::unique_ptr<Op>> LogSoftmaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<LogSoftmaxGradOp>(*this));
  return upops;
}

int64_t LogSoftmaxOp::getAxis() const {
  auto r = static_cast<int64_t>(inShape(getInIndex()).size());
  if (axis < -r || axis > r - 1) {
    throw error("LogSoftmax axis, {}, is outside of acceptable range [{}, {}]",
                axis,
                -r,
                r - 1);
  }

  if (axis < 0) {
    return r + axis;
  } else {
    return axis;
  }
}

void LogSoftmaxOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

std::vector<std::tuple<OperatorIdentifier, float>>
LogSoftmaxOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::LogSoftmaxInplace, 10}};
}

std::unique_ptr<Op>
LogSoftmaxOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::LogSoftmaxInplace) {
    return std::make_unique<LogSoftmaxInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

LogSoftmaxInplaceOp::LogSoftmaxInplaceOp(const LogSoftmaxOp &lsm_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::LogSoftmaxInplace,
                                lsm_op.getSettings()),
      axis(lsm_op.getAxis()) {}

std::unique_ptr<Op> LogSoftmaxInplaceOp::clone() const {
  return std::make_unique<LogSoftmaxInplaceOp>(*this);
}

void LogSoftmaxInplaceOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

LogSoftmaxGradOp::LogSoftmaxGradOp(const LogSoftmaxOp &op_)
    : Op(Onnx::GradOperators::LogSoftmaxGrad, op_.getSettings()),
      axis(op_.getAxis()) {}

std::unique_ptr<Op> LogSoftmaxGradOp::clone() const {
  return std::make_unique<LogSoftmaxGradOp>(*this);
}

const std::vector<GradInOutMapper> &LogSoftmaxGradOp::gradInputInfo() const {
  // input at index 0 (probGradInputIndex()) : gradient of output of logsoftmax
  // input at index 1 (actsIn()): input of logsoftmax (activations before p)
  // the (1-sparse) gradient of the output will be used
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradProbsInIndex(),
       LogSoftmaxOp::getOutIndex(),
       GradOpInType::GradOut},
      {getActsInIndex(), LogSoftmaxOp::getInIndex(), GradOpInType::In}};
  return inInfo;
}

const std::map<int, int> &LogSoftmaxGradOp::gradOutToNonGradIn() const {
  // the grad-op's output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), LogSoftmaxOp::getInIndex()}};
  return outInfo;
}

void LogSoftmaxGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradProbsInIndex());
}

void LogSoftmaxGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition logSoftmaxOpDef({OpDefinition::Inputs({{"input", T}}),
                                     OpDefinition::Outputs({{"output", T}}),
                                     OpDefinition::Attributes({
                                         {"axis", {"*"}},
                                     })});

static OpCreator<LogSoftmaxOp> logSoftmaxOpCreator(
    OpDefinitions({{Onnx::Operators::LogSoftmax_1, logSoftmaxOpDef},
                   {Onnx::Operators::LogSoftmax_11, logSoftmaxOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t axis = info.attributes.getAttribute<Attributes::Int>("axis", 1);

      return std::make_unique<LogSoftmaxOp>(info.opid, axis, info.settings);
    },
    true);

} // namespace

} // namespace popart
