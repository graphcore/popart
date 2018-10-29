#include <willow/averagepool.hpp>
#include <willow/error.hpp>
#include <willow/popx/averagepoolx.hpp>
#include <willow/util.hpp>

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

void AveragePoolGradOpx::grow() const {
  AveragePoolGradOp *agOp = getAveragePoolGradOp();
  AveragePoolOp *aOp = dynamic_cast<AveragePoolOp *>(agOp->getNonGradCreator());

  TensorId prePooledId  = inId(agOp->getPrePooledIn());
  TensorId pooledId     = inId(agOp->getPooledIn());
  TensorId gradPooledId = inId(agOp->getGradPooledIn());

  insert(outId(0),
         popnn::pooling::poolInputGradient(
             graph(),
             popnn::PoolingType::AVG, // poolingType
             aOp->spatialK_szt(),     // kernelShape
             aOp->strides_u32(),      // stride
             aOp->lowerPads_i32(),    // inputPaddingLower
             aOp->upperPads_i32(),
             get(prePooledId),  // in
             get(pooledId),     // pooled
             get(gradPooledId), // pooledGradient
             step(),            // prog
             idStr()            // debugPredix
             ));
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
