#include <memory>
#include <popart/op/subtract.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

SubtractOp::SubtractOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings_)
    : ElementWiseBinaryOp(_opid, settings_) {
  // TODO : Do not broadcast in version 6
}

std::unique_ptr<Op> SubtractOp::clone() const {
  return std::make_unique<SubtractOp>(*this);
}

std::vector<std::unique_ptr<Op>> SubtractOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_a0 = inShape(SubtractOp::getArg0InIndex());
  const auto &shape_o0 = outShape(SubtractOp::getOutIndex());

  upops.emplace_back(std::make_unique<SubtractArg0GradOp>(
      *this, npReductionAxis(shape_a0, shape_o0)));
  upops.emplace_back(std::make_unique<SubtractArg1GradOp>(*this));

  return upops;
}

SubtractArg0GradOp::SubtractArg0GradOp(const SubtractOp &op_,
                                       const std::vector<int64_t> &_axes)
    : ReduceSumOp(Onnx::GradOperators::SubArg0Grad,
                  _axes,
                  false,
                  op_.getSettings()),
      forward_op_arg_info(op_.inInfo(SubtractOp::getArg0InIndex())) {}

const std::map<int, int> &SubtractArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SubtractOp::getArg0InIndex()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SubtractOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

void SubtractArg0GradOp::setup() {
  outInfo(getOutIndex()) = forward_op_arg_info;
}

SubtractArg1GradOp::SubtractArg1GradOp(const SubtractOp &op_)
    : Op(Onnx::GradOperators::SubArg1Grad, op_.getSettings()),
      forward_op_arg_info(op_.inInfo(SubtractOp::getArg1InIndex())) {}

const std::map<int, int> &SubtractArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SubtractOp::getArg1InIndex()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractArg1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SubtractOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

std::unique_ptr<Op> SubtractArg1GradOp::clone() const {
  return std::make_unique<SubtractArg1GradOp>(*this);
}

void SubtractArg1GradOp::setup() {
  outInfo(getOutIndex()) = forward_op_arg_info;
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
