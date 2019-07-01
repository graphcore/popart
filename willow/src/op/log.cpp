#include <memory>
#include <poponnx/op/log.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

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
static OpCreator<LogOp> logOpCreator_6(Onnx::Operators::Log_6);
} // namespace

} // namespace poponnx
