#include <poponnx/makeunique.hpp>
#include <poponnx/op/sqrt.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SqrtOp::SqrtOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::unique_ptr<Op> SqrtOp::clone() const { return make_unique<SqrtOp>(*this); }

std::vector<std::unique_ptr<Op>> SqrtOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SqrtGradOp>(this));
  return upops;
}

void SqrtOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

SqrtGradOp::SqrtGradOp(SqrtOp *fwdOp)
    : Op({"SqrtGrad", fwdOp->pir, {}, getPoponnxDomain()}) {}

std::unique_ptr<Op> SqrtGradOp::clone() const {
  return make_unique<SqrtGradOp>(*this);
}

void SqrtGradOp::setup() { outInfo(getOutIndex()) = inInfo(getGradInIndex()); }

const std::vector<GradInOutMapper> &SqrtGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), 0, GradOpInType::GRADOUT},
      {getFwdOutInIndex(), SqrtOp::getOutIndex(), GradOpInType::OUT}};

  return inInfo;
}

const std::map<int, int> &SqrtGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SqrtOp::getInIndex()}};

  return outInfo;
}

} // namespace poponnx
