#include <poponnx/makeunique.hpp>
#include <poponnx/op/tanh.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

TanhOp::TanhOp(const OperatorIdentifier &_opid,
               Ir *_ir,
               const std::string &name,
               const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> TanhOp::clone() const { return make_unique<TanhOp>(*this); }

std::vector<std::unique_ptr<Op>> TanhOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<TanhGradOp>(this));
  return upops;
}

void TanhOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

TanhGradOp::TanhGradOp(TanhOp *fwdOp)
    : Op(Onnx::GradOperators::TanhGrad, fwdOp->pir) {}

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
static GradOpCreator<TanhGradOp>
    tanhGradOpCreator(Onnx::GradOperators::TanhGrad);
} // namespace

} // namespace poponnx
