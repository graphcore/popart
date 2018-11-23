#include <poponnx/error.hpp>
#include <poponnx/op/averagepool.hpp>
#include <poponnx/popx/op/averagepoolx.hpp>
#include <poponnx/util.hpp>

#include <popnn/Pooling.hpp>

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

void AveragePoolOpx::grow(poplar::program::Sequence &prog) const {
  AveragePoolOp *aOp = getAveragePoolOp();
  insert(outId(0),
         popnn::pooling::pool(graph(),
                              popnn::PoolingType::AVG,
                              aOp->spatialK_szt(),
                              aOp->strides_u32(),
                              aOp->lowerPads_i32(),
                              aOp->upperPads_i32(),
                              get(inId(0)),
                              prog,
                              idStr()));
}

void AveragePoolGradOpx::grow(poplar::program::Sequence &prog) const {
  AveragePoolGradOp *agOp  = getAveragePoolGradOp();
  const AveragePoolOp *aOp = agOp->getCloneOfCreator();

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
             prog,              // prog
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
