#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/maxpool.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/maxpoolx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popnn/Pooling.hpp>

namespace poponnx {
namespace popx {

MaxPoolOpx::MaxPoolOpx(Op *op, Devicex *devicex) : PoolOpx(op, devicex) {
  verifyOp<MaxPoolOp>(op, Onnx::Operators::MaxPool);
}

void MaxPoolOpx::grow(poplar::program::Sequence &prog) const {
  MaxPoolOp &aOp = getOp<MaxPoolOp>();

  auto pool_params = GetPoolingParameters(popnn::PoolingType::MAX,
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

void MaxPoolGradOpx::grow(poplar::program::Sequence &prog) const {
  MaxPoolGradOp &agOp  = getOp<MaxPoolGradOp>();
  const MaxPoolOp *aOp = agOp.getCloneOfCreator();

  TensorId prePooledId  = inId(agOp.getPrePooledInIndex());
  TensorId pooledId     = inId(agOp.getPooledInIndex());
  TensorId gradPooledId = inId(agOp.getGradPooledInIndex());

  auto pool_params = GetPoolingParameters(popnn::PoolingType::MAX,
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

MaxPoolGradOpx::MaxPoolGradOpx(Op *op, Devicex *devicex)
    : PoolOpx(op, devicex) {
  verifyOp<MaxPoolGradOp>(op, Onnx::GradOperators::MaxPoolGrad);
}

namespace {
OpxCreator<MaxPoolOpx> maxpoolOpxCreator(Onnx::Operators::MaxPool);
OpxCreator<MaxPoolGradOpx>
    maxpoolGradOpxCreator(Onnx::GradOperators::MaxPoolGrad);
} // namespace

} // namespace popx
} // namespace poponnx
