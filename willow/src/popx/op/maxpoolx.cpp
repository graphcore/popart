#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/maxpool.hpp>
#include <poponnx/popx/op/maxpoolx.hpp>

#include <popnn/Pooling.hpp>

namespace poponnx {
namespace popx {

MaxPoolOpx::MaxPoolOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::MAXPOOL) {
    throw error("cannot create MaxPoolOpx from " + op->op_type());
  }
}

MaxPoolOp *MaxPoolOpx::getMaxPoolOp() const {
  return dynamic_cast<MaxPoolOp *>(op_p);
}

void MaxPoolOpx::grow(poplar::program::Sequence &prog) const {
  MaxPoolOp *aOp = getMaxPoolOp();
  insert(outId(0),
         popnn::pooling::pool(graph(),
                              popnn::PoolingType::MAX,
                              aOp->spatialK_szt(),
                              aOp->strides_u32(),
                              aOp->lowerPads_i32(),
                              aOp->upperPads_i32(),
                              get(inId(0)),
                              prog,
                              idStr()));
}

void MaxPoolGradOpx::grow(poplar::program::Sequence &prog) const {
  MaxPoolGradOp *agOp  = getMaxPoolGradOp();
  const MaxPoolOp *aOp = agOp->getCloneOfCreator();

  TensorId prePooledId  = inId(agOp->getPrePooledInIndex());
  TensorId pooledId     = inId(agOp->getPooledInIndex());
  TensorId gradPooledId = inId(agOp->getGradPooledInIndex());

  insert(outId(0),
         popnn::pooling::poolInputGradient(
             graph(),
             popnn::PoolingType::MAX, // poolingType
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

MaxPoolGradOpx::MaxPoolGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::MAXPOOLGRAD) {
    throw error("cannot create MaxPoolGradOpx from " + op->op_type());
  }
}

MaxPoolGradOp *MaxPoolGradOpx::getMaxPoolGradOp() const {
  return dynamic_cast<MaxPoolGradOp *>(op_p);
}

} // namespace popx
} // namespace poponnx
