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
  // Give the gradient a slightly lower priority to help the scheduler
  priority -= 1;
}

std::unique_ptr<Op> SinGradOp::clone() const {
  return std::make_unique<SinGradOp>(*this);
}

namespace {
static OpCreator<SinOp> sinOpCreator(Onnx::Operators::Sin_7);
} // namespace

} // namespace popart
