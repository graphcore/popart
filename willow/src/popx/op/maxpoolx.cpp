#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/maxpool.hpp>
#include <poponnx/popx/op/maxpoolx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popnn/Pooling.hpp>

namespace poponnx {
namespace popx {

MaxPoolOpx::MaxPoolOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<MaxPoolOp>(op, Onnx::Operators::MaxPool);
}

void MaxPoolOpx::grow(poplar::program::Sequence &prog) const {
  MaxPoolOp &aOp = getOp<MaxPoolOp>();
  insert(outId(0),
         popnn::pooling::pool(graph(),
                              popnn::PoolingType::MAX,
                              aOp.spatialK_szt(),
                              aOp.strides_u32(),
                              aOp.lowerPads_i32(),
                              aOp.upperPads_i32(),
                              get(inId(0)),
                              prog,
                              idStr()));
}

void MaxPoolGradOpx::grow(poplar::program::Sequence &prog) const {
  MaxPoolGradOp &agOp  = getOp<MaxPoolGradOp>();
  const MaxPoolOp *aOp = agOp.getCloneOfCreator();

  TensorId prePooledId  = inId(agOp.getPrePooledInIndex());
  TensorId pooledId     = inId(agOp.getPooledInIndex());
  TensorId gradPooledId = inId(agOp.getGradPooledInIndex());

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
  verifyOp<MaxPoolGradOp>(op, Onnx::GradOperators::MaxPoolGrad);
}

namespace {
OpxCreator<MaxPoolOpx> maxpoolOpxCreator(Onnx::Operators::MaxPool);
OpxCreator<MaxPoolGradOpx>
    maxpoolGradOpxCreator(Onnx::GradOperators::MaxPoolGrad);
} // namespace

} // namespace popx
} // namespace poponnx
