#include <memory>
#include <vector>
#include <popart/op/less.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

LessOp::LessOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> LessOp::clone() const {
  return std::make_unique<LessOp>(*this);
}

std::vector<std::unique_ptr<Op>> LessOp::getGradOps() {
  throw error("PopART does not have a valid grad op corresponding to LessOp");
}

namespace {
static OpCreator<LessOp> LessOpCreator({Onnx::Operators::Less_7,
                                        Onnx::Operators::Less_9});
} // namespace

} // namespace popart
