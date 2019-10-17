#include <memory>
#include <popart/graph.hpp>
#include <popart/op/asin.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
AsinOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::AsinInplace, 10}};
}

std::unique_ptr<Op>
AsinOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::AsinInplace) {
    return std::make_unique<AsinInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

AsinInplaceOp::AsinInplaceOp(const AsinOp &asin_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::AsinInplace,
                                asin_op.getSettings()) {}

std::unique_ptr<Op> AsinInplaceOp::clone() const {
  return std::make_unique<AsinInplaceOp>(*this);
}

std::unique_ptr<Op> AsinOp::clone() const {
  return std::make_unique<AsinOp>(*this);
}

AsinOp::AsinOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::vector<std::unique_ptr<Op>> AsinOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<AsinGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> AsinGradOp::clone() const {
  return std::make_unique<AsinGradOp>(*this);
}

AsinGradOp::AsinGradOp(const AsinOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::AsinGrad, fwdop) {}

namespace {
static OpCreator<AsinOp> asinOpCreator(Onnx::Operators::Asin_7);
} // namespace

} // namespace popart
