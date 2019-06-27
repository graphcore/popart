#include <memory>
#include <vector>
#include <poponnx/op/less.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

LessOp::LessOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> LessOp::clone() const {
  return std::make_unique<LessOp>(*this);
}

std::vector<std::unique_ptr<Op>> LessOp::getGradOps() {
  throw error("PopONNX does not have a valid grad op corresponding to LessOp");
}

namespace {
static OpCreator<LessOp> LessOpCreator({Onnx::Operators::Less_7,
                                        Onnx::Operators::Less_9});
} // namespace

} // namespace poponnx
