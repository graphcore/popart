#include <memory>
#include <popart/op/sign.hpp>
#include <popart/opmanager.hpp>

namespace popart {

SignOp::SignOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::vector<std::unique_ptr<Op>> SignOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SignGradOp>(*this));
  return upops;
}

OperatorIdentifier SignOp::getOpId(const Ir &) {
  return Onnx::Operators::Sign_9;
}

std::unique_ptr<Op> SignOp::clone() const {
  return std::make_unique<SignOp>(*this);
}

SignGradOp::SignGradOp(const SignOp &op_)
    : Op(Onnx::GradOperators::SignGrad, op_.getSettings()) {}

const std::map<int, int> &SignGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SignOp::getInIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &SignGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SignOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

void SignGradOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

std::unique_ptr<Op> SignGradOp::clone() const {
  return std::make_unique<SignGradOp>(*this);
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
