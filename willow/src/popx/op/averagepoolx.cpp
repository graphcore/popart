#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/averagepool.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/averagepoolx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popnn/Pooling.hpp>

namespace poponnx {
namespace popx {

AveragePoolOpx::AveragePoolOpx(Op *op, Devicex *devicex)
    : PoolOpx(op, devicex) {
  verifyOp<AveragePoolOp>(
      op, {Onnx::Operators::AveragePool_1, Onnx::Operators::AveragePool_7});
}

void AveragePoolOpx::grow(poplar::program::Sequence &prog) const {
  AveragePoolOp &aOp = getOp<AveragePoolOp>();

  auto pool_params = GetPoolingParameters(popnn::PoolingType::AVG,
                                          op_p->inInfo(0),
                                          aOp.spatialK,
                                          aOp.strides,
                                          aOp.lowerPads(),
                                          aOp.upperPads());

  insert(outId(0),
         popnn::pooling::pool(graph(),
                              pool_params,
                              get(inId(0)),
                              prog,
                              idStr(),
                              dv_p->pooling_options));
}

void AveragePoolGradOpx::grow(poplar::program::Sequence &prog) const {
  AveragePoolGradOp &agOp  = getOp<AveragePoolGradOp>();
  const AveragePoolOp *aOp = agOp.getCloneOfCreator();

  TensorId prePooledId  = inId(agOp.getPrePooledInIndex());
  TensorId pooledId     = inId(agOp.getPooledInIndex());
  TensorId gradPooledId = inId(agOp.getGradPooledInIndex());

  auto pool_params = GetPoolingParameters(popnn::PoolingType::AVG,
                                          op_p->inInfo(0),
                                          aOp->spatialK,
                                          aOp->strides,
                                          aOp->lowerPads(),
                                          aOp->upperPads());

  insert(outId(0),
         popnn::pooling::poolInputGradient(graph(),
                                           pool_params,
                                           get(prePooledId),
                                           get(pooledId),
                                           get(gradPooledId),
                                           prog,
                                           idStr(),
                                           dv_p->pooling_options));
}

AveragePoolGradOpx::AveragePoolGradOpx(Op *op, Devicex *devicex)
    : PoolOpx(op, devicex) {
  verifyOp<AveragePoolGradOp>(op, Onnx::GradOperators::AveragePoolGrad);
}

namespace {
OpxCreator<AveragePoolOpx> averagePoolOpxCreator(
    {Onnx::Operators::AveragePool_1, Onnx::Operators::AveragePool_7});
OpxCreator<AveragePoolGradOpx>
    averagePoolGradOpxCreator(Onnx::GradOperators::AveragePoolGrad);
} // namespace

} // namespace popx
} // namespace poponnx
