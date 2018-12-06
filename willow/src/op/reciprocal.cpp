#include <poponnx/makeunique.hpp>
#include <poponnx/op/reciprocal.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReciprocalOp::ReciprocalOp(const OpConstructorBundle &bundle)
    : ElementWiseUnaryOp(bundle) {}

ReciprocalOp::ReciprocalOp(const onnx::NodeProto &node, Ir *ir)
    : ElementWiseUnaryOp(node, ir) {}

std::unique_ptr<Op> ReciprocalOp::clone() const {
  return make_unique<ReciprocalOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReciprocalOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  upops.emplace_back(make_unique<ReciprocalGradOp>(this));
  return upops;
}

ReciprocalGradOp::ReciprocalGradOp(ReciprocalOp *op_)
    : ElementWiseNonLinearUnaryGradOp(
          {"ReciprocalGrad", op_->pir, {}, getPoponnxDomain()}) {}

std::unique_ptr<Op> ReciprocalGradOp::clone() const {
  return make_unique<ReciprocalGradOp>(*this);
}

} // namespace poponnx
