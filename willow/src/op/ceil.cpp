#include <memory>
#include <poponnx/op/ceil.hpp>
#include <poponnx/opmanager.hpp>

namespace poponnx {

std::vector<std::tuple<OperatorIdentifier, float>>
CeilOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::CeilInplace, 10}};
}

CeilInplaceOp::CeilInplaceOp(const CeilOp &ceil_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::CeilInplace,
                                ceil_op.getSettings()) {}

std::unique_ptr<Op>
CeilOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::CeilInplace) {
    return std::make_unique<CeilInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

CeilOp::CeilOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> CeilOp::clone() const {
  return std::make_unique<CeilOp>(*this);
}

std::vector<std::unique_ptr<Op>> CeilOp::getGradOps() {
  throw error("PopONNX does not have a valid grad op corresponding to CeilOp");
}

std::unique_ptr<Op> CeilInplaceOp::clone() const {
  return std::make_unique<CeilInplaceOp>(*this);
}

namespace {
static OpCreator<CeilOp> ceilOpCreator({Onnx::Operators::Ceil_1,
                                        Onnx::Operators::Ceil_6});

} // namespace
} // namespace poponnx
