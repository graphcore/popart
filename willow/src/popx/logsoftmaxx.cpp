#include <willow/error.hpp>
#include <willow/logsoftmax.hpp>
#include <willow/popx/logsoftmaxx.hpp>

namespace willow {
namespace popx {

LogSoftmaxOpx::LogSoftmaxOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::LOGSOFTMAX) {
    throw error("cannot create LogSoftmaxOpx from " + op->op_type());
  }
}

LogSoftmaxOp *LogSoftmaxOpx::getLogSoftmaxOp() const {
  return dynamic_cast<LogSoftmaxOp *>(getOp());
}

LogSoftmaxGradOpx::LogSoftmaxGradOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::LOGSOFTMAXGRAD) {
    throw error("cannot create LogSoftmaxGradOpx from " + op->op_type());
  }
}

LogSoftmaxGradOp *LogSoftmaxGradOpx::getLogSoftmaxGradOp() const {
  return dynamic_cast<LogSoftmaxGradOp *>(getOp());
}

} // namespace popx
} // namespace willow
