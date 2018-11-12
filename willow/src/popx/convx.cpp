#include <poponnx/conv.hpp>
#include <poponnx/error.hpp>
#include <poponnx/popx/convx.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

namespace willow {
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

std::vector<TensorId> ConvOpx::mustExistBeforeCreate(int) const {
  // creation of both weights and of input are done
  // without requiring the pre-existance of any
  // other poplar::Tensor, so returning empty TensorId vector
  return {};
}

void ConvOpx::grow() const {

  auto outTensor = poplin::convolution(
      graph(),                           // graph
      get(getConvOp()->dataIn()->id),    // in
      get(getConvOp()->weightsIn()->id), // weights
      fwdParams,                         // params
      false,                             // transposeAndFlipWeights,
      dv_p->progs.step(),                // prog
      idStr(),                           // debugPrefix
      enigma::toPoplibsConvOptions(dv_p->fwdConvOptions), // options
      &dv_p->convCache                                    // cache
  );

  insert(outId(0), outTensor);
}

void ConvDataGradOpx::grow() const {
  ConvDataGradOp *gradOp = getConvDataGradOp();
  auto outTensor         = poplin::convolution(
      graph(),                                 // graph
      get(inId(gradOp->getGradConvolvedIn())), // in
      get(inId(gradOp->getWeightsIn())),       // weights
      dataGradParams,                          // params
      true,               // transposeAndFlipWeights,
      dv_p->progs.step(), // prog
      idStr(),            // debugPrefix
      enigma::toPoplibsConvOptions(dv_p->bwdConvOptions), // options
      &dv_p->convCache                                    // cache
  );

  insert(outId(0), outTensor);
}

void ConvWeightsGradOpx::grow() const {
  ConvWeightsGradOp *gradOp = getConvWeightsGradOp();
  const ConvOp *convOp      = gradOp->getCloneOfCreator();
  poplar::Tensor wGrad      = poplin::calculateWeightDeltas(
      graph(),                                           // graph
      get(inId(gradOp->getGradConvolvedIn())),           // zDeltas,
      get(inId(gradOp->getPreConvolvedIn())),            // activations,
      getFwdConvParams(convOp),                          // params
      step(),                                            // prog
      idStr(),                                           // debugPrefix
      enigma::toPoplibsConvOptions(dv_p->wuConvOptions), // options
      &dv_p->convCache);                                 // cache

  insert(outId(0), wGrad);
}

ConvOpx::ConvOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {

  if (op->opType != OpType::CONV) {
    throw error("cannot create ConvOpx from " + op->op_type());
  }

  ConvOp *cOp = getConvOp();
  if (cOp->dataIn()->info.rank() != 4 || cOp->weightsIn()->info.rank() != 4) {
    throw error("Poplar only supports convolutions with 2 spatial dimensions");
  }

  fwdParams = getFwdConvParams(cOp);
}

bool ConvOpx::createsEquiv(int ind0, Opx *opx1, int ind1) const {
  // if opx1 is not a ConvOpx, it does not create the same poplar::Tensor
  if (opx1->op_p->opType != OpType::CONV) {
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

ConvOp *ConvOpx::getConvOp() const { return dynamic_cast<ConvOp *>(op_p); }

bool ConvOpx::canCreateInput(int) const { return true; }

poplar::Tensor ConvOpx::createInput(int index) const {

  if (index == convWeightsInIndex()) {
    return poplin::createWeights(
        graph(),                                            // graph
        fwdParams,                                          // params
        op_p->str(),                                        // name
        enigma::toPoplibsConvOptions(dv_p->fwdConvOptions), // options
        &dv_p->convCache                                    // cache
    );
  } else if (index == convDataInIndex()) {
    return poplin::createInput(
        graph(),                                            // graph
        fwdParams,                                          // params
        idStr(),                                            // name
        enigma::toPoplibsConvOptions(dv_p->fwdConvOptions), // options
        &dv_p->convCache                                    // cache
    );
  } else {
    throw error("conv opx cannot create tensor at this index yet");
  }
}

ConvDataGradOpx::ConvDataGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::CONVDATAGRAD) {
    throw error("cannot create ConvDataGradOpx from " + op->op_type());
  }
  dataGradParams = getDataGradParams(getConvDataGradOp());
}

ConvDataGradOp *ConvDataGradOpx::getConvDataGradOp() const {
  return dynamic_cast<ConvDataGradOp *>(op_p);
}

ConvWeightsGradOpx::ConvWeightsGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op->opType != OpType::CONVWEIGHTSGRAD) {
    throw error("cannot create ConvWeightsGradOpx from " + op->op_type());
  }
}

ConvWeightsGradOp *ConvWeightsGradOpx::getConvWeightsGradOp() const {
  return dynamic_cast<ConvWeightsGradOp *>(op_p);
}

} // namespace popx
} // namespace willow
