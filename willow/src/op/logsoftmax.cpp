#include <memory>
#include <poponnx/error.hpp>
#include <poponnx/op/logsoftmax.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

LogSoftmaxOp::LogSoftmaxOp(const OperatorIdentifier &_opid,
                           const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> LogSoftmaxOp::clone() const {
  return std::make_unique<LogSoftmaxOp>(*this);
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
