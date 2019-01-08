#include <poponnx/makeunique.hpp>
#include <poponnx/op/log.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

LogOp::LogOp(const OperatorIdentifier &_opid,
             Ir *_ir,
             const std::string &name,
             const Attributes &_attr)
    : ElementWiseUnaryOp(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> LogOp::clone() const { return make_unique<LogOp>(*this); }

std::vector<std::unique_ptr<Op>> LogOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<LogGradOp>(this));
  return upops;
}

LogGradOp::LogGradOp(LogOp *fwdOp)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::LogGrad,
                                      fwdOp->pir) {}

std::unique_ptr<Op> LogGradOp::clone() const {
  return make_unique<LogGradOp>(*this);
}

namespace {
static OpCreator<LogOp> logOpCreator(Onnx::Operators::Log);
static GradOpCreator<LogGradOp> logGradOpCreator(Onnx::GradOperators::LogGrad);
} // namespace

} // namespace poponnx
