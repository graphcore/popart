#include <poponnx/makeunique.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SinOp::SinOp(const OpConstructorBundle &bundle) : ElementWiseUnaryOp(bundle) {}

SinOp::SinOp(const onnx::NodeProto &node, Ir *_pir)
    : ElementWiseUnaryOp(node, _pir) {}

std::unique_ptr<Op> SinOp::clone() const { return make_unique<SinOp>(*this); }

std::vector<std::unique_ptr<Op>> SinOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SinGradOp>(this));
  return upops;
}

SinGradOp::SinGradOp(SinOp *fwdOp)
    : ElementWiseNonLinearUnaryGradOp(
          {"SinGrad", fwdOp->pir, {}, getPoponnxDomain()}) {}

std::unique_ptr<Op> SinGradOp::clone() const {
  return make_unique<SinGradOp>(*this);
}

} // namespace poponnx
