#include <vector>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/equal.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

EqualOp::EqualOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> EqualOp::clone() const {
  return make_unique<EqualOp>(*this);
}

std::vector<std::unique_ptr<Op>> EqualOp::getGradOps() {
  throw error("PopONNX does not have a valid grad op corresponding to EqualOp");
}

namespace {
static OpCreator<EqualOp> EqualOpCreator({Onnx::Operators::Equal_1,
                                          Onnx::Operators::Equal_7});
} // namespace

} // namespace poponnx
