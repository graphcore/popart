#include <algorithm>
#include <memory>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/shrink.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

ShrinkOp::ShrinkOp(const OperatorIdentifier &opid_,
                   float lambd,
                   float bias,
                   const Op::Settings &opSettings)
    : ElementWiseUnaryOp(opid, opSettings), lambd_(lambd), bias_(bias) {}

std::unique_ptr<Op> ShrinkOp::clone() const {
  return std::make_unique<ShrinkOp>(*this);
}

std::vector<std::unique_ptr<Op>> ShrinkOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<ShrinkGradOp>(*this));
  return result;
}

std::vector<std::tuple<OperatorIdentifier, float>>
ShrinkOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::ShrinkInplace, 10}};
}

std::unique_ptr<Op>
ShrinkOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ShrinkInplace) {
    return std::make_unique<ShrinkInplaceOp>(*this);
  }
  return Op::getInplaceVariant(operator_id);
}

void ShrinkOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("lambd", lambd_);
  os.appendAttribute("bias", bias_);
}

ShrinkInplaceOp::ShrinkInplaceOp(const ShrinkOp &op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::ShrinkInplace,
                                op.getSettings()),
      lambd_(op.lambd()), bias_(op.bias()) {}

std::unique_ptr<Op> ShrinkInplaceOp::clone() const {
  return std::make_unique<ShrinkInplaceOp>(*this);
}

void ShrinkInplaceOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("lambd", lambd_);
  os.appendAttribute("bias", bias_);
}

ShrinkGradOp::ShrinkGradOp(const ShrinkOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::ShrinkGrad, fwdop),
      lambd_(fwdop.lambd()), bias_(fwdop.bias()) {}

std::unique_ptr<Op> ShrinkGradOp::clone() const {
  return std::make_unique<ShrinkGradOp>(*this);
}

void ShrinkGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("lambd", lambd_);
  os.appendAttribute("bias", bias_);
}

namespace {
static OpCreator<ShrinkOp> shrinkOpCreator(
    {Onnx::Operators::Shrink_9},
    [](const OperatorIdentifier &opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      float lambd = attr.getAttribute<Attributes::Float>("lambd", 0.5f);
      float bias  = attr.getAttribute<Attributes::Float>("bias", 0.0f);

      return std::unique_ptr<Op>(new ShrinkOp(opid, lambd, bias, settings));
    },
    true);

} // namespace
} // namespace popart
