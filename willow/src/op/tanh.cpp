#include <poponnx/makeunique.hpp>
#include <poponnx/op/tanh.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

TanhOp::TanhOp(const OpConstructorBundle &bundle) : Op(bundle) {}

TanhOp::TanhOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::unique_ptr<Op> TanhOp::clone() const { return make_unique<TanhOp>(*this); }

std::vector<std::unique_ptr<Op>> TanhOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<TanhGradOp>(this));
  return upops;
}

void TanhOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

TanhGradOp::TanhGradOp(TanhOp *fwdOp)
    : Op({OpType::TANHGRAD, fwdOp->pir, {}}) {}

std::unique_ptr<Op> TanhGradOp::clone() const {
  return make_unique<TanhGradOp>(*this);
}

const std::vector<GradInOutMapper> &TanhGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), TanhOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdArgInIndex(), TanhOp::getInIndex(), GradOpInType::IN}};

  return inInfo;
}

const std::map<int, int> &TanhGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), TanhOp::getInIndex()}};

  return outInfo;
}

void TanhGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getFwdArgInIndex());
}

} // namespace poponnx
