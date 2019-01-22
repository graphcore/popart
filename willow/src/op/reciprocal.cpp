#include <poponnx/makeunique.hpp>
#include <poponnx/op/reciprocal.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReciprocalOp::ReciprocalOp(const OperatorIdentifier &_opid,
                           const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> ReciprocalOp::clone() const {
  return make_unique<ReciprocalOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReciprocalOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  upops.emplace_back(make_unique<ReciprocalGradOp>(*this));
  return upops;
}

ReciprocalGradOp::ReciprocalGradOp(const ReciprocalOp &op_)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::ReciprocalGrad,
                                      op_) {}

std::unique_ptr<Op> ReciprocalGradOp::clone() const {
  return make_unique<ReciprocalGradOp>(*this);
}

namespace {
static OpCreator<ReciprocalOp>
    receiprocalOpCreator(Onnx::Operators::Reciprocal_6);

} // namespace

} // namespace poponnx
