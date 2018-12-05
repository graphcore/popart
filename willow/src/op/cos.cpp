#include <poponnx/makeunique.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

CosOp::CosOp(const OpConstructorBundle &bundle) : Op(bundle) {}

CosOp::CosOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::unique_ptr<Op> CosOp::clone() const { return make_unique<CosOp>(*this); }

std::vector<std::unique_ptr<Op>> CosOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<CosGradOp>(this));
  return upops;
}

void CosOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

CosGradOp::CosGradOp(CosOp *fwdOp)
    : Op({"CosGrad", fwdOp->pir, {}, getPoponnxDomain()}) {}

std::unique_ptr<Op> CosGradOp::clone() const {
  return make_unique<CosGradOp>(*this);
}

const std::vector<GradInOutMapper> &CosGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), 0, GradOpInType::GRADOUT},
      {getFwdArgInIndex(), CosOp::getInIndex(), GradOpInType::IN}};

  return inInfo;
}

const std::map<int, int> &CosGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), CosOp::getInIndex()}};

  return outInfo;
}

void CosGradOp::setup() { outInfo(getOutIndex()) = inInfo(getFwdArgInIndex()); }

} // namespace poponnx
