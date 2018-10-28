#include <willow/error.hpp>
#include <willow/logsoftmax.hpp>
#include <willow/popx/logsoftmaxx.hpp>

namespace willow {
namespace popx {

LogSoftmaxOpx::LogSoftmaxOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::LOGSOFTMAX) {
    throw error("cannot create LogSoftmaxOpx from " + op->op_type());
  }
}

void LogSoftmaxOpx::grow() const {
  throw error("LogSoftmaxOpx::grow not implemented yet");
}

LogSoftmaxOp *LogSoftmaxOpx::getLogSoftmaxOp() const {
  return dynamic_cast<LogSoftmaxOp *>(op_p);
}

LogSoftmaxGradOpx::LogSoftmaxGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op->opType != OpType::LOGSOFTMAXGRAD) {
    throw error("cannot create LogSoftmaxGradOpx from " + op->op_type());
  }
}

LogSoftmaxGradOp *LogSoftmaxGradOpx::getLogSoftmaxGradOp() const {
  return dynamic_cast<LogSoftmaxGradOp *>(op_p);
}

LogSoftmaxGradDirectOpx::LogSoftmaxGradDirectOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op->opType != OpType::LOGSOFTMAXGRADDIRECT) {
    throw error("cannot create LogSoftmaxGradDirectOpx from " + op->op_type());
  }
}

LogSoftmaxGradDirectOp *
LogSoftmaxGradDirectOpx::getLogSoftmaxGradDirectOp() const {
  return dynamic_cast<LogSoftmaxGradDirectOp *>(op_p);
}

} // namespace popx
} // namespace willow
