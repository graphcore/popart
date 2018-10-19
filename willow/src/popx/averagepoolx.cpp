#include <willow/averagepool.hpp>
#include <willow/error.hpp>
#include <willow/popx/averagepoolx.hpp>

namespace willow {
namespace popx {

AveragePoolOpx::AveragePoolOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::AVERAGEPOOL) {
    throw error("cannot create AveragePoolOpx from " + op->op_type());
  }
}

AveragePoolOp *AveragePoolOpx::getAveragePoolOp() const {
  return dynamic_cast<AveragePoolOp *>(getOp());
}

} // namespace popx
} // namespace willow
