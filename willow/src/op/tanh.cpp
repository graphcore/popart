#include <poponnx/makeunique.hpp>
#include <poponnx/op/tanh.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

TanhOp::TanhOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> TanhOp::clone() const { return make_unique<TanhOp>(*this); }

std::vector<std::unique_ptr<Op>> TanhOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<TanhGradOp>(*this));
  return upops;
}

void TanhOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

TanhGradOp::TanhGradOp(const TanhOp &fwdOp)
    : Op(Onnx::GradOperators::TanhGrad, fwdOp.getSettings()) {}

std::unique_ptr<Op> TanhGradOp::clone() const {
  return make_unique<TanhGradOp>(*this);
}

const std::vector<GradInOutMapper> &TanhGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), TanhOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdOutInIndex(), TanhOp::getOutIndex(), GradOpInType::OUT}};

  return inInfo;
}

const std::map<int, int> &TanhGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), TanhOp::getInIndex()}};

  return outInfo;
}

void TanhGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getFwdOutInIndex());
}

namespace {
static OpCreator<TanhOp> tanhOpCreator(Onnx::Operators::Tanh_6);

} // namespace

} // namespace poponnx
