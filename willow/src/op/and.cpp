#include <vector>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/and.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

AndOp::AndOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> AndOp::clone() const { return make_unique<AndOp>(*this); }

std::vector<std::unique_ptr<Op>> AndOp::getGradOps() {
  throw error("PopONNX does not have a valid grad op corresponding to AndOp");
}

namespace {
static OpCreator<AndOp> AndOpCreator({Onnx::Operators::And_1,
                                      Onnx::Operators::And_7});
} // namespace

} // namespace poponnx
