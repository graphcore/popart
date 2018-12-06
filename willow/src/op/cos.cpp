#include <poponnx/makeunique.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

CosOp::CosOp(const OpConstructorBundle &bundle) : ElementWiseUnaryOp(bundle) {}

CosOp::CosOp(const onnx::NodeProto &node, Ir *_pir)
    : ElementWiseUnaryOp(node, _pir) {}

std::unique_ptr<Op> CosOp::clone() const { return make_unique<CosOp>(*this); }

std::vector<std::unique_ptr<Op>> CosOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<CosGradOp>(this));
  return upops;
}

CosGradOp::CosGradOp(CosOp *fwdOp)
    : ElementWiseNonLinearUnaryGradOp(
          {"CosGrad", fwdOp->pir, {}, getPoponnxDomain()}) {}

std::unique_ptr<Op> CosGradOp::clone() const {
  return make_unique<CosGradOp>(*this);
}

} // namespace poponnx
