#include <poponnx/makeunique.hpp>
#include <poponnx/op/abs.hpp>
#include <poponnx/opmanager.hpp>

namespace poponnx {

AbsOp::AbsOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> AbsOp::clone() const { return make_unique<AbsOp>(*this); }

std::vector<std::unique_ptr<Op>> AbsOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<AbsGradOp>(*this));
  return upops;
}

void AbsOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

AbsGradOp::AbsGradOp(const AbsOp &op_)
    : Op(Onnx::GradOperators::AbsGrad, op_.getSettings()) {}

const std::map<int, int> &AbsGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AbsOp::getInIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AbsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), AbsOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdArgInIndex(), AbsOp::getInIndex(), GradOpInType::IN}};
  return inInfo;
}

void AbsGradOp::setup() { outInfo(getOutIndex()) = inInfo(getGradInIndex()); }

namespace {
static OpCreator<AbsOp> absOpCreator({Onnx::Operators::Abs_6});
} // namespace

} // namespace poponnx
