#include <poponnx/makeunique.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {
SinOp::SinOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> SinOp::clone() const { return make_unique<SinOp>(*this); }

std::vector<std::unique_ptr<Op>> SinOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SinGradOp>(*this));
  return upops;
}

SinGradOp::SinGradOp(const SinOp &fwdOp)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::SinGrad, fwdOp) {}

std::unique_ptr<Op> SinGradOp::clone() const {
  return make_unique<SinGradOp>(*this);
}

namespace {
static OpCreator<SinOp> sinOpCreator(Onnx::Operators::Sin_7);
} // namespace

} // namespace poponnx
