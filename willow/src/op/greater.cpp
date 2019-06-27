#include <memory>
#include <vector>
#include <poponnx/op/greater.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

GreaterOp::GreaterOp(const OperatorIdentifier &_opid,
                     const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> GreaterOp::clone() const {
  return std::make_unique<GreaterOp>(*this);
}

std::vector<std::unique_ptr<Op>> GreaterOp::getGradOps() {
  throw error(
      "PopONNX does not have a valid grad op corresponding to GreaterOp");
}

namespace {
static OpCreator<GreaterOp> GreaterOpCreator({Onnx::Operators::Greater_7,
                                              Onnx::Operators::Greater_9});
} // namespace

} // namespace poponnx
