#include <poponnx/makeunique.hpp>
#include <poponnx/op/exp.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

std::vector<std::tuple<OperatorIdentifier, float>>
ExpOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::ExpInplace, 10}};
}

std::unique_ptr<Op>
ExpOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ExpInplace) {
    return make_unique<ExpInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

ExpInplaceOp::ExpInplaceOp(const ExpOp &exp_op)
    : Op(Onnx::CustomOperators::ExpInplace, exp_op.getSettings()) {}

void ExpInplaceOp::setup() {
  // no output, nothing to setup
  outInfo(ExpOp::getOutIndex()) = inInfo(ExpOp::getInIndex());
}

ExpOp::ExpOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> ExpOp::clone() const { return make_unique<ExpOp>(*this); }

std::vector<std::unique_ptr<Op>> ExpOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ExpGradOp>(*this));
  return upops;
}

void ExpOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

ExpGradOp::ExpGradOp(const ExpOp &fwdOp)
    : Op(Onnx::GradOperators::ExpGrad, fwdOp.getSettings()) {}

std::unique_ptr<Op> ExpGradOp::clone() const {
  return make_unique<ExpGradOp>(*this);
}

const std::vector<GradInOutMapper> &ExpGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), ExpOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdOutInIndex(), ExpOp::getOutIndex(), GradOpInType::OUT}};

  return inInfo;
}

const std::map<int, int> &ExpGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ExpOp::getInIndex()}};

  return outInfo;
}

void ExpGradOp::setup() { outInfo(getOutIndex()) = inInfo(getFwdOutInIndex()); }

namespace {
static OpCreator<ExpOp> expOpCreator(Onnx::Operators::Exp_6);
} // namespace

} // namespace poponnx
