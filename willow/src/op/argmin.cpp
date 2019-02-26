#include <poponnx/makeunique.hpp>
#include <poponnx/op/argmin.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

std::unique_ptr<Op> ArgMinOp::clone() const {
  return make_unique<ArgMinOp>(*this);
}

namespace {
std::unique_ptr<Op> argMinFactory(const OperatorIdentifier &_opid,
                                  const Op::Settings &settings,
                                  const Attributes &attr) {
  int64_t axis     = attr.getAttribute<Attributes::Int>("axis", 0);
  int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);

  return make_unique<ArgMinOp>(_opid, axis, keepdims, settings);
}

static OpCreator<ArgMinOp>
    argMinOpCreator(Onnx::Operators::ArgMin_1, argMinFactory, true);
} // namespace

} // namespace poponnx
