#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/dropout.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

DropoutOp::DropoutOp(const OperatorIdentifier &_opid,
                     float ratio_,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), ratio(ratio_) {}

std::unique_ptr<Op> DropoutOp::clone() const {
  return make_unique<DropoutOp>(*this);
}

void DropoutOp::setup() {

  if (getIr().getExecutionMode() == Ir::ExecutionMode::TRAINING) {
    throw error("Dropout does not support training");
  }

  outInfo(getOutIndex()) = inInfo(getInIndex());
}

std::vector<std::unique_ptr<Op>> DropoutOp::getGradOps() {
  throw error("DropoutOp should be removed by pattern 'OpToIdentity' before "
              "call to getGradOps");
}

void DropoutOp::appendAttributes(std::stringstream &ss,
                                 const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "ratio", ratio);
}

// Dropout in testing mode can be replaced by the identity
bool DropoutOp::canBeReplacedByIdentity() {
  return (getIr().isTesting() || getIr().isEvaulation());
}

namespace {
static OpCreator<DropoutOp> dropoutOpCreator(
    {Onnx::Operators::Dropout_6, Onnx::Operators::Dropout_7},
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      float ratio = attr.getAttribute<Attributes::Float>("ratio", 0.5f);

      return std::unique_ptr<Op>(new DropoutOp(_opid, ratio, settings));
    },
    true);

} // namespace
} // namespace poponnx
