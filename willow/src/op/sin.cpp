#include <poponnx/makeunique.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {
SinOp::SinOp(const OperatorIdentifier &_opid,
             Ir *_ir,
             const std::string &name,
             const Attributes &_attr)
    : ElementWiseUnaryOp(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> SinOp::clone() const { return make_unique<SinOp>(*this); }

std::vector<std::unique_ptr<Op>> SinOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SinGradOp>(this));
  return upops;
}

SinGradOp::SinGradOp(SinOp *fwdOp)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::SinGrad,
                                      fwdOp->pir) {}

std::unique_ptr<Op> SinGradOp::clone() const {
  return make_unique<SinGradOp>(*this);
}

namespace {
static OpCreator<SinOp> sinOpCreator(Onnx::Operators::Sin);
static GradOpCreator<SinGradOp> sinGradOpCreator(Onnx::GradOperators::SinGrad);
} // namespace

} // namespace poponnx
