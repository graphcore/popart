#include <willow/averagepool.hpp>
#include <willow/error.hpp>
#include <willow/popx/averagepoolx.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <popnn/Pooling.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

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

void AveragePoolOpx::grow() const {
  AveragePoolOp *aOp = getAveragePoolOp();
  insert(outId(0),
         popnn::pooling::pool(graph(),
                              popnn::PoolingType::AVG,
                              aOp->spatialK_szt(),
                              aOp->strides_u32(),
                              aOp->lowerPads_i32(),
                              aOp->upperPads_i32(),
                              get(inId(0)),
                              step(),
                              idStr()));
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
