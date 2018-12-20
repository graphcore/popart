#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/averagepool.hpp>
#include <poponnx/popx/op/averagepoolx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popnn/Pooling.hpp>

namespace poponnx {
namespace popx {

AveragePoolOpx::AveragePoolOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<AveragePoolOp>(op, Onnx::Operators::AveragePool);
}

void AveragePoolOpx::grow(poplar::program::Sequence &prog) const {

  AveragePoolOp &aOp = getOp<AveragePoolOp>();

  insert(outId(0),
         popnn::pooling::pool(graph(),
                              popnn::PoolingType::AVG,
                              aOp.spatialK_szt(),
                              aOp.strides_u32(),
                              aOp.lowerPads_i32(),
                              aOp.upperPads_i32(),
                              get(inId(0)),
                              prog,
                              idStr()));
}

void AveragePoolGradOpx::grow(poplar::program::Sequence &prog) const {
  AveragePoolGradOp &agOp  = getOp<AveragePoolGradOp>();
  const AveragePoolOp *aOp = agOp.getCloneOfCreator();

  TensorId prePooledId  = inId(agOp.getPrePooledInIndex());
  TensorId pooledId     = inId(agOp.getPooledInIndex());
  TensorId gradPooledId = inId(agOp.getGradPooledInIndex());

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
  verifyOp<AveragePoolGradOp>(op, Onnx::GradOperators::AveragePoolGrad);
}

namespace {
OpxCreator<AveragePoolOpx> averagePoolOpxCreator(Onnx::Operators::AveragePool);
OpxCreator<AveragePoolGradOpx>
    averagePoolGradOpxCreator(Onnx::GradOperators::AveragePoolGrad);
} // namespace

} // namespace popx
} // namespace poponnx
