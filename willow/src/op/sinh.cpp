#include <memory>
#include <popart/graph.hpp>
#include <popart/op/sinh.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
SinhOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::SinhInplace, 10}};
}

std::unique_ptr<Op>
SinhOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::SinhInplace) {
    return std::make_unique<SinhInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

SinhInplaceOp::SinhInplaceOp(const SinhOp &sinh_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::SinhInplace,
                                sinh_op.getSettings()) {}

std::unique_ptr<Op> SinhInplaceOp::clone() const {
  return std::make_unique<SinhInplaceOp>(*this);
}

std::unique_ptr<Op> SinhOp::clone() const {
  return std::make_unique<SinhOp>(*this);
}

SinhOp::SinhOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::vector<std::unique_ptr<Op>> SinhOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SinhGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> SinhGradOp::clone() const {
  return std::make_unique<SinhGradOp>(*this);
}

SinhGradOp::SinhGradOp(const SinhOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::SinhGrad, fwdop) {}

namespace {
static OpCreator<SinhOp> sinhOpCreator(Onnx::Operators::Sinh_9);
} // namespace

} // namespace popart
