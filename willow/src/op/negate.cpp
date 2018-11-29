#include <poponnx/op/negate.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

NegateOp::NegateOp(const OpConstructorBundle &bundle) : Op(bundle) {}

NegateOp::NegateOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::unique_ptr<Op> NegateOp::clone() const {
  return make_unique<NegateOp>(*this);
}

std::vector<std::unique_ptr<Op>> NegateOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<NegateGradOp>(this));
  return upops;
}

void NegateOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

NegateGradOp::NegateGradOp(NegateOp *fwdOp)
    : NegateOp({"NegateGrad", fwdOp->pir, {}, getPoponnxDomain()}) {}

std::unique_ptr<Op> NegateGradOp::clone() const {
  return make_unique<NegateGradOp>(*this);
}

const std::vector<GradInOutMapper> &NegateGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), NegateOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &NegateGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), NegateOp::getInIndex()}};

  return outInfo;
}

} // namespace poponnx
