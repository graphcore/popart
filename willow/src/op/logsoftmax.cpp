#include <memory>
#include <poponnx/error.hpp>
#include <poponnx/op/logsoftmax.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

LogSoftmaxOp::LogSoftmaxOp(const OperatorIdentifier &_opid,
                           int64_t axis_,
                           const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_), axis(axis_) {}

std::unique_ptr<Op> LogSoftmaxOp::clone() const {
  return std::make_unique<LogSoftmaxOp>(*this);
}

std::vector<std::unique_ptr<Op>> LogSoftmaxOp::getGradOps() {
  throw error("LogSoftmaxOp should be removed by pattern 'LogSoftmaxOp' before "
              "call to getGradOps");
}

namespace {
static OpCreator<LogSoftmaxOp> logSoftmaxOpCreator(
    Onnx::Operators::LogSoftmax_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t axis = attr.getAttribute<Attributes::Int>("axis", 1);

      return std::make_unique<LogSoftmaxOp>(_opid, axis, settings);
    },
    true);

} // namespace

} // namespace poponnx
