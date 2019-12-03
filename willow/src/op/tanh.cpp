#include <memory>
#include <popart/op/tanh.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

TanhOp::TanhOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> TanhOp::clone() const {
  return std::make_unique<TanhOp>(*this);
}

std::vector<std::unique_ptr<Op>> TanhOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<TanhGradOp>(*this));
  return upops;
}

TanhGradOp::TanhGradOp(const TanhOp &fwdOp)
    : Op(Onnx::GradOperators::TanhGrad, fwdOp.getSettings()) {}

std::unique_ptr<Op> TanhGradOp::clone() const {
  return std::make_unique<TanhGradOp>(*this);
}

const std::vector<GradInOutMapper> &TanhGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), TanhOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdOutInIndex(), TanhOp::getOutIndex(), GradOpInType::OUT}};

  return inInfo;
}

const std::map<int, int> &TanhGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), TanhOp::getInIndex()}};

  return outInfo;
}

void TanhGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getFwdOutInIndex());
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition tanhOpDef({OpDefinition::Inputs({
                                   {"input", T},
                               }),
                               OpDefinition::Outputs({{"output", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<TanhOp> tanhOpCreator(OpDefinitions({
    {Onnx::Operators::Tanh_6, tanhOpDef},
}));

} // namespace

} // namespace popart
