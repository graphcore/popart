#include <algorithm>
#include <vector>

#include <memory>
#include <poponnx/op/argmax.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

std::unique_ptr<Op> ArgMaxOp::clone() const {
  return std::make_unique<ArgMaxOp>(*this);
}

namespace {
std::unique_ptr<Op> argMaxFactory(const OperatorIdentifier &_opid,
                                  const Op::Settings &settings,
                                  const Attributes &attr) {
  int64_t axis     = attr.getAttribute<Attributes::Int>("axis", 0);
  int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);

  return std::make_unique<ArgMaxOp>(_opid, axis, keepdims, settings);
}

static OpCreator<ArgMaxOp>
    ArgMaxOpCreator(Onnx::Operators::ArgMax_1, argMaxFactory, true);
} // namespace

} // namespace poponnx
