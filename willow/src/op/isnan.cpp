#include <memory>
#include <popart/ir.hpp>
#include <popart/op/isnan.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

IsNaN::IsNaN(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryBooleanOp(_opid, settings_) {}

std::unique_ptr<Op> IsNaN::clone() const {
  return std::make_unique<IsNaN>(*this);
}

OperatorIdentifier IsNaN::getOpId(const Ir &) {
  return Onnx::Operators::IsNaN_9;
}

namespace {
static OpCreator<IsNaN> IsNaNCreator(Onnx::Operators::IsNaN_9);
} // namespace

} // namespace popart
