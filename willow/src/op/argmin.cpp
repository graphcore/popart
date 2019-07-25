#include <memory>
#include <popart/op/argmin.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::unique_ptr<Op> ArgMinOp::clone() const {
  return std::make_unique<ArgMinOp>(*this);
}

namespace {
std::unique_ptr<Op> argMinFactory(const OperatorIdentifier &_opid,
                                  const Op::Settings &settings,
                                  const Attributes &attr) {
  int64_t axis     = attr.getAttribute<Attributes::Int>("axis", 0);
  int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);

  return std::make_unique<ArgMinOp>(_opid, axis, keepdims, settings);
}

static OpCreator<ArgMinOp>
    argMinOpCreator(Onnx::Operators::ArgMin_1, argMinFactory, true);
} // namespace

} // namespace popart
