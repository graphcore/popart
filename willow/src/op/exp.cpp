#include <poponnx/makeunique.hpp>
#include <poponnx/op/exp.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ExpOp::ExpOp(const OperatorIdentifier &_opid,
             Ir *_ir,
             const std::string &name,
             const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> ExpOp::clone() const { return make_unique<ExpOp>(*this); }

std::vector<std::unique_ptr<Op>> ExpOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ExpGradOp>(this));
  return upops;
}

void ExpOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

ExpGradOp::ExpGradOp(ExpOp *fwdOp)
    : Op(Onnx::GradOperators::ExpGrad, fwdOp->pir) {}

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

namespace {
static OpCreator<ExpOp> expOpCreator(Onnx::Operators::Exp_6);
static GradOpCreator<ExpGradOp> expGradOpCreator(Onnx::GradOperators::ExpGrad);
} // namespace

} // namespace poponnx
