#include <memory>
#include <poponnx/ir.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

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
static OpCreator<CosOp> cosOpCreator(Onnx::Operators::Cos_7);
} // namespace

} // namespace poponnx
