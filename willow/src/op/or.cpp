#include <memory>
#include <vector>
#include <poponnx/op/or.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

OrOp::OrOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> OrOp::clone() const {
  return std::make_unique<OrOp>(*this);
}

std::vector<std::unique_ptr<Op>> OrOp::getGradOps() {
  throw error("PopONNX does not have a valid grad op corresponding to OrOp");
}

namespace {
static OpCreator<OrOp> OrOpCreator({Onnx::Operators::Or_1,
                                    Onnx::Operators::Or_7});
} // namespace

} // namespace poponnx
