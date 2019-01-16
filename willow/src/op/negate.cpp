#include <poponnx/op/negate.hpp>

#include <poponnx/makeunique.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

NegateOp::NegateOp(const OperatorIdentifier &_opid,
                   Ir *_ir,
                   const std::string &name,
                   const Attributes &_attr)
    : ElementWiseUnaryOp(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> NegateOp::clone() const {
  return make_unique<NegateOp>(*this);
}

std::vector<std::unique_ptr<Op>> NegateOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<NegateGradOp>(this));
  return upops;
}

NegateGradOp::NegateGradOp(NegateOp *fwdOp)
    : NegateOp(Onnx::GradOperators::NegGrad, fwdOp->pir) {}

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

namespace {
static OpCreator<NegateOp> negateOpCreator(Onnx::Operators::Neg_6);
static GradOpCreator<NegateGradOp>
    negateGradOpCreator(Onnx::GradOperators::NegGrad);
} // namespace

} // namespace poponnx
