#include <memory>
#include <popart/ir.hpp>
#include <popart/op/isinf.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

IsInf::IsInf(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryBooleanOp(_opid, settings_) {}

std::unique_ptr<Op> IsInf::clone() const {
  return std::make_unique<IsInf>(*this);
}

OperatorIdentifier IsInf::getOpId(const Ir &) {
  return Onnx::Operators::IsInf_10;
}

namespace {
static OpCreator<IsInf> IsInfCreator(Onnx::Operators::IsInf_10);
} // namespace

} // namespace popart
