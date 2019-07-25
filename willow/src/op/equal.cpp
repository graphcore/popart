#include <memory>
#include <vector>
#include <popart/op/equal.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

EqualOp::EqualOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> EqualOp::clone() const {
  return std::make_unique<EqualOp>(*this);
}

std::vector<std::unique_ptr<Op>> EqualOp::getGradOps() {
  throw error("PopART does not have a valid grad op corresponding to EqualOp");
}

namespace {
static OpCreator<EqualOp> EqualOpCreator({Onnx::Operators::Equal_1,
                                          Onnx::Operators::Equal_7});
} // namespace

} // namespace popart
