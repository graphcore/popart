#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/popx/convoptionsx.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/graphcachex.hpp>
#include <poponnx/popx/op/convx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>

namespace poponnx {
namespace popx {

poplin::ConvParams getFwdConvParams(const ConvOp *cOp) {

  std::vector<unsigned> zeros(cOp->nSpatialDims, 0);
  std::vector<bool> falses(cOp->nSpatialDims, false);
  std::vector<unsigned> ones(cOp->nSpatialDims, 1);

  // we assume that the output type is the same as the input
  auto popOutType = cOp->outType;

  return poplin::ConvParams(popType(popOutType), // dType,
                            cOp->batchSize,      // batchSize,
                            cOp->spatialD_szt(), // inputFieldShape,
                            cOp->spatialK_szt(), // kernelShape,

                            cOp->nInChans,       // inputChannels,
                            cOp->getNOutChans(), // outputChannels,
                            cOp->group,          // numConvGroups,

                            zeros,                // inputTruncationLower,
                            zeros,                // inputTruncationUpper,
                            ones,                 // inputDilation,
                            cOp->lowerPads_u32(), // inputPaddingLower,
                            cOp->upperPads_u32(), // inputPaddingUpper
                            falses,               // flipInput,

                            zeros,                // kernelTruncationLower,
                            zeros,                // kernelTruncationUpper,
                            cOp->dilations_u32(), // kernelDilation,
                            zeros,                // kernelPaddingLower,
                            zeros,                // kernelPaddingUpper,
                            falses,               // flipKernel,

                            zeros,              // outputTruncationLower,
                            zeros,              // outputTruncationUpper,
                            cOp->strides_u32(), // stride,
                            zeros,              // outputPaddingLower,
                            zeros               // outputPaddingUpper.
  );
}

poplin::ConvParams getDataGradParams(const ConvDataGradOp *convDataGradOp) {
  // we get the fwd params, and then use a utility
  // function to convert to bwd params.
  auto fwdParams = getFwdConvParams(convDataGradOp->getCloneOfCreator());
  // this utility function converts fwd params to bwd params.
  // see poplin/ConvUtil.hpp
  return poplin::getGradientParams(fwdParams);
}

const poplin::ConvParams &ConvOpx::getParams() const { return fwdParams; }

std::vector<TensorId> ConvOpx::mustExistBeforeCreate(InIndex) const {
  // creation of both weights and of input are done
  // without requiring the pre-existance of any
  // other poplar::Tensor, so returning empty TensorId vector
  return {};
}

void ConvOpx::grow(poplar::program::Sequence &prog) const {
  ConvOp &convOp = getOp<ConvOp>();
  auto outTensor =
      dv_p->graphCache.convolution(graph(),                     // graph
                                   get(convOp.dataIn()->id),    // in
                                   get(convOp.weightsIn()->id), // weights
                                   fwdParams,                   // params
                                   false, // transposeAndFlipWeights,
                                   prog,  // prog
                                   convOp.cacheOperation, // cacheOperation
                                   idStr(),               // debugPrefix
                                   dv_p->fwdConvOptions,  // options
                                   &dv_p->convCache       // cache
      );

  insert(outId(0), outTensor);
}

void ConvDataGradOpx::grow(poplar::program::Sequence &prog) const {
  ConvDataGradOp &gradOp = getOp<ConvDataGradOp>();
  const ConvOp *convOp   = gradOp.getCloneOfCreator();
  auto outTensor         = dv_p->graphCache.convolution(
      graph(),                                     // graph
      get(inId(gradOp.getGradConvolvedInIndex())), // in
      get(inId(gradOp.getWeightsInIndex())),       // weights
      dataGradParams,                              // params
      true,                   // transposeAndFlipWeights,
      prog,                   // prog
      convOp->cacheOperation, // cacheOperation
      idStr(),                // debugPrefix
      dv_p->bwdConvOptions,   // options
      &dv_p->convCache        // cache
  );

  insert(outId(0), outTensor);
}

void ConvWeightsGradOpx::grow(poplar::program::Sequence &prog) const {
  ConvWeightsGradOp &gradOp = getOp<ConvWeightsGradOp>();
  const ConvOp *convOp      = gradOp.getCloneOfCreator();
  poplar::Tensor wGrad      = dv_p->graphCache.calculateWeightDeltas(
      graph(),                                     // graph
      get(inId(gradOp.getGradConvolvedInIndex())), // zDeltas,
      get(inId(gradOp.getPreConvolvedInIndex())),  // activations,
      getFwdConvParams(convOp),                    // params
      prog,                                        // prog
      convOp->cacheOperation,                      // cacheOperation
      idStr(),                                     // debugPrefix
      dv_p->wuConvOptions,                         // options
      &dv_p->convCache);                           // cache

  insert(outId(0), wGrad);
}

ConvOpx::ConvOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ConvOp>(op, Onnx::Operators::Conv);

  ConvOp &cOp = getOp<ConvOp>();
  if (cOp.dataIn()->info.rank() != 4 || cOp.weightsIn()->info.rank() != 4) {
    throw error("Poplar only supports convolutions with 2 spatial dimensions");
  }

  fwdParams = getFwdConvParams(&cOp);
}

bool ConvOpx::createsEquiv(int ind0, Opx *opx1, int ind1) const {
  // if opx1 is not a ConvOpx, it does not create the same poplar::Tensor
  if (opx1->op_p->opid != Onnx::Operators::Conv) {
    return false;
  }

  // if opx1 (which we now know is ConvOpx) would create the tensor at
  // a different input index, creation is not equivalent
  if (ind0 != ind1) {
    return false;
  }

  // finally, check that the convolution parameters are the same
  ConvOpx *rhs = dynamic_cast<ConvOpx *>(opx1);
  if (getParams() != rhs->getParams()) {
    return false;
  }

  return true;
}

bool ConvOpx::canCreateInput(InIndex) const { return true; }

poplar::Tensor ConvOpx::createInput(InIndex index) const {

  if (index == ConvOp::getWeightsInIndex()) {
    return poplin::createWeights(
        graph(),                              // graph
        fwdParams,                            // params
        op_p->str(),                          // name
        dv_p->fwdConvOptions.toOptionFlags(), // options
        &dv_p->convCache                      // cache
    );
  } else if (index == ConvOp::getDataInIndex()) {
    return poplin::createInput(graph(),                              // graph
                               fwdParams,                            // params
                               idStr(),                              // name
                               dv_p->fwdConvOptions.toOptionFlags(), // options
                               &dv_p->convCache                      // cache
    );
  } else {
    throw error("conv opx cannot create tensor at this index yet");
  }
}

ConvDataGradOpx::ConvDataGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ConvDataGradOp>(op, Onnx::GradOperators::ConvDataGrad);
  dataGradParams = getDataGradParams(&(getOp<ConvDataGradOp>()));
}

ConvWeightsGradOpx::ConvWeightsGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ConvWeightsGradOp>(op, Onnx::GradOperators::ConvWeightsGrad);
}

namespace {
OpxCreator<ConvOpx> convpxCreator(Onnx::Operators::Conv);
OpxCreator<ConvDataGradOpx>
    convDataGradOpxCreator(Onnx::GradOperators::ConvDataGrad);
OpxCreator<ConvWeightsGradOpx>
    convWeightsGradOpxCreator(Onnx::GradOperators::ConvWeightsGrad);
} // namespace

} // namespace popx
} // namespace poponnx
