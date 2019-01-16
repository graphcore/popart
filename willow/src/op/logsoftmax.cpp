#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/logsoftmax.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

LogSoftmaxOp::LogSoftmaxOp(const OperatorIdentifier &_opid,
                           Ir *_ir,
                           const std::string &name,
                           const Attributes &_attr)
    : ElementWiseUnaryOp(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> LogSoftmaxOp::clone() const {
  return make_unique<LogSoftmaxOp>(*this);
}

std::vector<std::unique_ptr<Op>> LogSoftmaxOp::getGradOps() {
  throw error("LogSoftmaxOp should be removed by pattern 'LogSoftmaxOp' before "
              "call to getGradOps");
}

namespace {
static OpCreator<LogSoftmaxOp>
    logSoftmaxOpCreator(Onnx::Operators::LogSoftmax_1);
} // namespace

} // namespace poponnx
