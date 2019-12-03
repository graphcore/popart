#include <memory>
#include <popart/ir.hpp>
#include <popart/op/cos.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

CosOp::CosOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> CosOp::clone() const {
  return std::make_unique<CosOp>(*this);
}

std::vector<std::unique_ptr<Op>> CosOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<CosGradOp>(*this));
  return upops;
}

OperatorIdentifier CosOp::getOpId(const Ir &) { return Onnx::Operators::Cos_7; }

CosGradOp::CosGradOp(const CosOp &fwdOp)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::CosGrad, fwdOp) {}

std::unique_ptr<Op> CosGradOp::clone() const {
  return std::make_unique<CosGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition cosOpDef({OpDefinition::Inputs({
                                  {"input", T},
                              }),
                              OpDefinition::Outputs({{"output", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<CosOp>
    cosOpCreator(OpDefinitions({{Onnx::Operators::Cos_7, cosOpDef}}));

} // namespace

} // namespace popart
