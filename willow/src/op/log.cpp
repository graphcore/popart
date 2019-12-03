#include <memory>
#include <popart/op/log.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

LogOp::LogOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> LogOp::clone() const {
  return std::make_unique<LogOp>(*this);
}

std::vector<std::unique_ptr<Op>> LogOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<LogGradOp>(*this));
  return upops;
}

LogGradOp::LogGradOp(const LogOp &fwdOp)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::LogGrad, fwdOp) {}

std::unique_ptr<Op> LogGradOp::clone() const {
  return std::make_unique<LogGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition logOpDef({OpDefinition::Inputs({{"input", T}}),
                              OpDefinition::Outputs({{"output", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<LogOp>
    logOpCreator_6(OpDefinitions({{Onnx::Operators::Log_6, logOpDef}}));
} // namespace

} // namespace popart
