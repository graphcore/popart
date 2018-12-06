#include <poponnx/makeunique.hpp>
#include <poponnx/op/exp.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ExpOp::ExpOp(const OpConstructorBundle &bundle) : Op(bundle) {}

ExpOp::ExpOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::unique_ptr<Op> ExpOp::clone() const { return make_unique<ExpOp>(*this); }

std::vector<std::unique_ptr<Op>> ExpOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ExpGradOp>(this));
  return upops;
}

void ExpOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

ExpGradOp::ExpGradOp(ExpOp *fwdOp)
    : Op({"ExpGrad", fwdOp->pir, {}, getPoponnxDomain()}) {}

std::unique_ptr<Op> ExpGradOp::clone() const {
  return make_unique<ExpGradOp>(*this);
}

const std::vector<GradInOutMapper> &ExpGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), ExpOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdOutInIndex(), ExpOp::getOutIndex(), GradOpInType::OUT}};

  return inInfo;
}

const std::map<int, int> &ExpGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ExpOp::getInIndex()}};

  return outInfo;
}

void ExpGradOp::setup() { outInfo(getOutIndex()) = inInfo(getFwdOutInIndex()); }

} // namespace poponnx
