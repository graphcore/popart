#include <willow/error.hpp>
#include <willow/popx/softmaxx.hpp>
#include <willow/softmax.hpp>

namespace willow {
namespace popx {

SoftmaxOpx::SoftmaxOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SOFTMAX) {
    throw error("cannot create SoftmaxOpx from " + op->op_type());
  }
}

void SoftmaxOpx::grow() const {
  throw error("SoftmaxOpx::grow not implemented yet");
}

SoftmaxOp *SoftmaxOpx::getSoftmaxOp() const {
  return dynamic_cast<SoftmaxOp *>(op_p);
}

SoftmaxGradOpx::SoftmaxGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SOFTMAXGRAD) {
    throw error("cannot create SoftmaxGradOpx from " + op->op_type());
  }
}

SoftmaxGradOp *SoftmaxGradOpx::getSoftmaxGradOp() const {
  return dynamic_cast<SoftmaxGradOp *>(op_p);
}

SoftmaxGradDirectOpx::SoftmaxGradDirectOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op->opType != OpType::SOFTMAXGRADDIRECT) {
    throw error("cannot create SoftmaxGradDirectOpx from " + op->op_type());
  }
}

SoftmaxGradDirectOp *SoftmaxGradDirectOpx::getSoftmaxGradDirectOp() const {
  return dynamic_cast<SoftmaxGradDirectOp *>(op_p);
}

} // namespace popx
} // namespace willow
