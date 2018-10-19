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

} // namespace popx
} // namespace willow
