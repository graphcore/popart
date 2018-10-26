#include <willow/averagepool.hpp>
#include <willow/error.hpp>
#include <willow/popx/averagepoolx.hpp>

namespace willow {
namespace popx {

AveragePoolOpx::AveragePoolOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::AVERAGEPOOL) {
    throw error("cannot create AveragePoolOpx from " + op->op_type());
  }
}

AveragePoolOp *AveragePoolOpx::getAveragePoolOp() const {
  return dynamic_cast<AveragePoolOp *>(op_p);
}

AveragePoolGradOpx::AveragePoolGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op->opType != OpType::AVERAGEPOOLGRAD) {
    throw error("cannot create AveragePoolGradOpx from " + op->op_type());
  }
}

AveragePoolGradOp *AveragePoolGradOpx::getAveragePoolGradOp() const {
  return dynamic_cast<AveragePoolGradOp *>(op_p);
}

} // namespace popx
} // namespace willow
