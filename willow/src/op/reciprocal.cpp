#include <poponnx/makeunique.hpp>
#include <poponnx/op/reciprocal.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReciprocalOp::ReciprocalOp(const OperatorIdentifier &_opid,
                           Ir *_ir,
                           const std::string &name,
                           const Attributes &_attr)
    : ElementWiseUnaryOp(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> ReciprocalOp::clone() const {
  return make_unique<ReciprocalOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReciprocalOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  upops.emplace_back(make_unique<ReciprocalGradOp>(this));
  return upops;
}

ReciprocalGradOp::ReciprocalGradOp(ReciprocalOp *op_)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::ReciprocalGrad,
                                      op_->pir) {}

std::unique_ptr<Op> ReciprocalGradOp::clone() const {
  return make_unique<ReciprocalGradOp>(*this);
}

namespace {
static OpCreator<ReciprocalOp>
    receiprocalOpCreator(Onnx::Operators::Reciprocal_6);
static GradOpCreator<ReciprocalGradOp>
    receiprocalGradOpCreator(Onnx::GradOperators::ReciprocalGrad);
} // namespace

} // namespace poponnx
