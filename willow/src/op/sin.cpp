#include <memory>
#include <popart/op/sin.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
SinOp::SinOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> SinOp::clone() const {
  return std::make_unique<SinOp>(*this);
}

std::vector<std::unique_ptr<Op>> SinOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SinGradOp>(*this));
  return upops;
}

SinGradOp::SinGradOp(const SinOp &fwdOp)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::SinGrad, fwdOp) {
}

std::unique_ptr<Op> SinGradOp::clone() const {
  return std::make_unique<SinGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition sinOpDef({OpDefinition::Inputs({{"input", T}}),
                              OpDefinition::Outputs({{"output", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<SinOp> sinOpCreator(OpDefinitions({
    {Onnx::Operators::Sin_7, sinOpDef},
}));
} // namespace

} // namespace popart
