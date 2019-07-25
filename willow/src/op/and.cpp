#include <memory>
#include <vector>
#include <popart/op/and.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

AndOp::AndOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> AndOp::clone() const {
  return std::make_unique<AndOp>(*this);
}

std::vector<std::unique_ptr<Op>> AndOp::getGradOps() {
  throw error("PopART does not have a valid grad op corresponding to AndOp");
}

namespace {
static OpCreator<AndOp> AndOpCreator({Onnx::Operators::And_1,
                                      Onnx::Operators::And_7});
} // namespace

} // namespace popart
