#include <poponnx/makeunique.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SinOp::SinOp(const OpConstructorBundle &bundle) : Op(bundle) {}

SinOp::SinOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::unique_ptr<Op> SinOp::clone() const { return make_unique<SinOp>(*this); }

std::vector<std::unique_ptr<Op>> SinOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SinGradOp>(this));
  return upops;
}

void SinOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

SinGradOp::SinGradOp(SinOp *fwdOp)
    : Op({"SinGrad", fwdOp->pir, {}, getPoponnxDomain()}) {}

std::unique_ptr<Op> SinGradOp::clone() const {
  return make_unique<SinGradOp>(*this);
}

const std::vector<GradInOutMapper> &SinGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), 0, GradOpInType::GRADOUT},
      {getFwdArgInIndex(), SinOp::getInIndex(), GradOpInType::IN}};

  return inInfo;
}

const std::map<int, int> &SinGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SinOp::getInIndex()}};

  return outInfo;
}

void SinGradOp::setup() { outInfo(getOutIndex()) = inInfo(getFwdArgInIndex()); }

} // namespace poponnx
