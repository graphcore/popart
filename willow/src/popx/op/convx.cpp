#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/convx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/popx/poplaroptionsx.hpp>
#include <poponnx/tensor.hpp>

#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>

namespace poponnx {

namespace popx {

/*
static void printPoplinConvParams(const poplin::ConvParams &params) {
  logging::pattern::info(
      "poplin::ConvParams t:{} bs:{} s:{} ks:{} c:{},{} g:{} i:({} {} {} {} {} "
      "{}) k:({} {} {} {} {} {}) o:({} {} {} {} {})",
      params.dType,
      params.batchSize,
      params.inputFieldShape,
      params.kernelShape,
      params.inputChannels,
      params.outputChannels,
      params.numConvGroups,
      params.inputTransform.truncationLower,
      params.inputTransform.truncationUpper,
      params.inputTransform.dilation,
      params.inputTransform.paddingLower,
      params.inputTransform.paddingUpper,
      vXtoY<bool, int>(params.inputTransform.flip),
      params.kernelTransform.truncationLower,
      params.kernelTransform.truncationUpper,
      params.kernelTransform.dilation,
      params.kernelTransform.paddingLower,
      params.kernelTransform.paddingUpper,
      vXtoY<bool, int>(params.kernelTransform.flip),
      params.outputTransform.truncationLower,
      params.outputTransform.truncationUpper,
      params.outputTransform.stride,
      params.outputTransform.paddingLower,
      params.outputTransform.paddingUpper);
}
*/

static poplin::ConvParams getPoplarConvParams(const ConvParameters &param) {
  return poplin::ConvParams(
      popType(param.type),
      param.batchSize,
      vXtoY<int64_t, size_t>(param.inputShape),
      vXtoY<int64_t, size_t>(param.kernelShape),
      param.numInChannels,
      param.numOutChannels,
      param.numGroups,

      vXtoY<int64_t, unsigned>(param.inputTransformation.lowerTruncation),
      vXtoY<int64_t, unsigned>(param.inputTransformation.upperTruncation),
      vXtoY<int64_t, unsigned>(param.inputTransformation.dilation),
      vXtoY<int64_t, unsigned>(param.inputTransformation.lowerPadding),
      vXtoY<int64_t, unsigned>(param.inputTransformation.upperPadding),
      param.inputTransformation.flip,

      vXtoY<int64_t, unsigned>(param.kernelTransformation.lowerTruncation),
      vXtoY<int64_t, unsigned>(param.kernelTransformation.upperTruncation),
      vXtoY<int64_t, unsigned>(param.kernelTransformation.dilation),
      vXtoY<int64_t, unsigned>(param.kernelTransformation.lowerPadding),
      vXtoY<int64_t, unsigned>(param.kernelTransformation.upperPadding),
      param.kernelTransformation.flip,

      vXtoY<int64_t, unsigned>(param.outputTransformation.lowerTruncation),
      vXtoY<int64_t, unsigned>(param.outputTransformation.upperTruncation),
      vXtoY<int64_t, unsigned>(param.outputTransformation.stride),
      vXtoY<int64_t, unsigned>(param.outputTransformation.lowerPadding),
      vXtoY<int64_t, unsigned>(param.outputTransformation.upperPadding));
}

static ConvParameters
convertPoplarConvParameters(const poplin::ConvParams &popParams) {

  ConvParameters params;
  params.batchSize     = popParams.batchSize;
  params.inputShape    = vXtoY<std::size_t, int64_t>(popParams.inputFieldShape);
  params.kernelShape   = vXtoY<std::size_t, int64_t>(popParams.kernelShape);
  params.numInChannels = popParams.getNumInputChans();
  params.numOutChannels = popParams.getNumOutputChans();
  params.numGroups      = popParams.getNumConvGroups();

  auto convertInput = [](ConvParameters::Input &input,
                         const poplin::ConvParams::InputTransform &popInput) {
    input.lowerTruncation = vXtoY<unsigned, int64_t>(popInput.truncationLower);
    input.upperTruncation = vXtoY<unsigned, int64_t>(popInput.truncationUpper);
    input.dilation        = vXtoY<unsigned, int64_t>(popInput.dilation);
    input.lowerPadding    = vXtoY<unsigned, int64_t>(popInput.paddingLower);
    input.upperPadding    = vXtoY<unsigned, int64_t>(popInput.paddingUpper);
    input.flip            = popInput.flip;
  };

  convertInput(params.inputTransformation, popParams.inputTransform);
  convertInput(params.kernelTransformation, popParams.kernelTransform);

  auto convertOutput =
      [](ConvParameters::Output &output,
         const poplin::ConvParams::OutputTransform &popOutput) {
        output.lowerTruncation =
            vXtoY<unsigned, int64_t>(popOutput.truncationLower);
        output.upperTruncation =
            vXtoY<unsigned, int64_t>(popOutput.truncationUpper);
        output.stride       = vXtoY<unsigned, int64_t>(popOutput.stride);
        output.lowerPadding = vXtoY<unsigned, int64_t>(popOutput.paddingLower);
        output.upperPadding = vXtoY<unsigned, int64_t>(popOutput.paddingUpper);
      };

  convertOutput(params.outputTransformation, popParams.outputTransform);

  return params;
}

ConvParameters getConvGradParameters(const ConvParameters &fwdParams) {

  // Let us cheat for now use poplar
  poplin::ConvParams popBwdParams =
      poplin::getGradientParams(getPoplarConvParams(fwdParams));

  ConvParameters bwdParams = convertPoplarConvParameters(popBwdParams);
  bwdParams.type           = fwdParams.type;

  return bwdParams;
}

ConvParameters canonicalizeConvParams(const ConvParameters &param) {
  poplin::ConvParams popParams = popx::getPoplarConvParams(param);

  auto canonicalizedPopParams = poplin::canonicalizeParams(popParams);

  ConvParameters result = convertPoplarConvParameters(canonicalizedPopParams);
  result.type           = param.type;
  return result;
}

std::vector<TensorId> ConvOpx::mustExistBeforeCreate(InIndex) const {
  // creation of both weights and of input are done
  // without requiring the pre-existance of any
  // other poplar::Tensor, so returning empty TensorId vector
  return {};
}

// If user provides 4D weights (missing 'group' dimension), add
// an outer dimension, size 1
static poplar::Tensor
addGroupDimensionIfMissing(const poplar::Tensor &weights) {
  poplar::Tensor weights5D = weights;
  if (weights.rank() == 4) {
    weights5D = weights.expand({0});
  }
  return weights5D;
}

void ConvOpx::grow(poplar::program::Sequence &prog) const {

  ConvOp &op          = getOp<ConvOp>();
  const auto &in      = getInTensor(ConvOp::getDataInIndex());
  const auto &weights = getInTensor(ConvOp::getWeightsInIndex());

  poplar::Tensor weights5D = addGroupDimensionIfMissing(weights);

  // Work out the option based on the phase of the op
  // Conv can be bwd depending on the phase.
  PoplarOptions *options = nullptr;
  if (op.getPhase() == Phase::FWD) {
    options = &dv_p->fwdConvOptions;
  } else if (op.getPhase() == Phase::BWD) {
    options = &dv_p->bwdConvOptions;
  } else {
    throw error("Unexpected phase {} for conv",
                static_cast<int>(op.getPhase()));
  }

  poplin::ConvParams popConvParams = getPoplarConvParams(op.getParameters());

  auto outTensor = poplin::convolution(graph(),
                                       in,
                                       weights5D,
                                       popConvParams,
                                       false,
                                       prog,
                                       idStr(),
                                       options->toOptionFlags(),
                                       &(dv_p->convCache));

  setOutTensor(ConvOp::getOutIndex(), outTensor);
}

void ConvWeightsGradOpx::grow(poplar::program::Sequence &prog) const {
  ConvWeightsGradOp &gradOp = getOp<ConvWeightsGradOp>();
  const ConvOp *convOp      = gradOp.getCloneOfCreator();

  const poplar::Tensor &zDelta =
      getInTensor(ConvWeightsGradOp::getGradConvolvedInIndex());
  const poplar::Tensor &activations =
      getInTensor(ConvWeightsGradOp::getPreConvolvedInIndex());

  poplar::Tensor wGrad = poplin::calculateWeightDeltas(
      graph(),
      zDelta,
      activations,
      getPoplarConvParams(convOp->getParameters()),
      prog,
      idStr(),
      dv_p->wuConvOptions.toOptionFlags(),
      &dv_p->convCache);

  // Shape of weights Poponnx Tensor of forward Op
  // auto fwdShape = convOp->inInfo(convOp->getWeightsInIndex()).shape_szt(); //
  // segfault
  auto fwdShape = gradOp.outInfo(ConvWeightsGradOp::getOutIndex()).shape_szt();

  // If shapes disagree only on first (grouping) dimension, as in
  //   IR shape:             [   a, b, c, d]
  //   poplar::Tensor shape: [1, a, b, c, d]
  // then squeeze grouping dimension from poplar::Tensor
  if (wGrad.rank() == 5 && fwdShape.size() == 4) {
    auto wGradShape = wGrad.shape();
    if (std::equal(
            wGradShape.begin() + 1, wGradShape.end(), fwdShape.begin()) &&
        wGradShape[0] == 1) {
      wGrad = wGrad.squeeze({0});
    }
  }

  setOutTensor(ConvWeightsGradOp::getOutIndex(), wGrad);
}

ConvOpx::ConvOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ConvOp>(op, Onnx::Operators::Conv_1);

  ConvOp &cOp = getOp<ConvOp>();
  if (cOp.dataIn()->info.rank() != 4 || cOp.weightsIn()->info.rank() != 4) {
    throw error("Poplar only supports convolutions with 2 spatial dimensions");
  }
}

bool ConvOpx::createsEquiv(int ind0, Opx *opx1, int ind1) const {
  // if opx1 is not a ConvOpx, it does not create the same poplar::Tensor
  if (opx1->op_p->opid != Onnx::Operators::Conv_1) {
    return false;
  }

  // if opx1 (which we now know is ConvOpx) would create the tensor at
  // a different input index, creation is not equivalent
  if (ind0 != ind1) {
    return false;
  }

  // finally, check that the convolution parameters are the same
  auto &lhsOp = getOp<ConvOp>();

  ConvOpx *rhs = dynamic_cast<ConvOpx *>(opx1);
  auto &rhsOp  = rhs->getOp<ConvOp>();
  if (lhsOp.getParameters() != rhsOp.getParameters()) {
    return false;
  }

  return true;
}

InputCreatorType ConvOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANCREATE;
}

poplar::Tensor ConvOpx::createInput(InIndex index,
                                    const std::string &name) const {

  auto &op = getOp<ConvOp>();

  if (index == ConvOp::getWeightsInIndex()) {
    poplar::Tensor input =
        poplin::createWeights(graph(),                                 // graph
                              getPoplarConvParams(op.getParameters()), // params
                              name,                                    // name
                              dv_p->fwdConvOptions.toOptionFlags(), // options
                              &dv_p->convCache                      // cache
        );

    // If the user supplies a 4D weights tensor as input to conv,
    // createWeights returns 5D tensor, with outer 'group' dim = 1
    //
    // This is not robust in the case where we unwind the weights tensor
    // to the input. The unwind functions shouldn't all have to support
    // this particular case where the allocator candidate is conv.
    //
    // So if we want to support the case where the user's input shape results
    // in a 4D weight tensor, then we need to squeeze the 0th dimension from
    // the tensor returned from createWeights:
    if (input.rank() == 5 && op_p->inRank(index) == 4) {
      // If shapes disagree only on first dimension, as in
      //   IR shape :            [   a, b, c, d]
      //   poplar::Tensor shape: [1, a, b, c, d]
      auto ptshape = input.shape();
      auto irshape = op_p->inInfo(index).shape_szt();
      if (std::equal(ptshape.begin() + 1, ptshape.end(), irshape.begin()) &&
          ptshape[0] == 1) {
        input = input.squeeze({0});
      }
    }
    return input;
  } else if (index == ConvOp::getDataInIndex()) {
    return poplin::createInput(
        graph(),                                 // graph
        getPoplarConvParams(op.getParameters()), // params
        name,                                    // name
        dv_p->fwdConvOptions.toOptionFlags(),    // options
        &dv_p->convCache                         // cache
    );
  } else {
    throw error("conv opx cannot create tensor at this index yet");
  }
}

ConvWeightsGradOpx::ConvWeightsGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ConvWeightsGradOp>(op, Onnx::GradOperators::ConvWeightsGrad);
}

ConvFlipWeightsGradOpx::ConvFlipWeightsGradOpx(Op *op_, Devicex *devicex_)
    : Opx(op_, devicex_) {
  verifyOp<ConvFlipWeightsOp>(op_, Onnx::CustomOperators::ConvFlipWeights);
}

void ConvFlipWeightsGradOpx::grow(poplar::program::Sequence &seq) const {

  auto &op = getOp<ConvFlipWeightsOp>();

  poplar::Tensor weights   = getInTensor(ConvFlipWeightsOp::getInIndex());
  poplar::Tensor weights5D = addGroupDimensionIfMissing(weights);

  auto fwdOptions            = dv_p->bwdConvOptions;
  fwdOptions.options["pass"] = "TRAINING_FWD";

  poplin::ConvParams popConvParams = getPoplarConvParams(op.getParameters());

  auto convWeights =
      poplin::createWeights(graph(),
                            popConvParams,
                            inTensor(ConvFlipWeightsOp::getInIndex())->str() +
                                sNameDelimiter + "flipped",
                            fwdOptions.toOptionFlags(),
                            &dv_p->convCache);

  poplin::weightsTransposeChansFlipXY(
      graph(), weights5D, convWeights, seq, debugPrefix("transposeXY"));

  // Taken the 1 off the front convWeights if it was added.
  if (weights.rank() != weights5D.rank()) {
    convWeights = convWeights.squeeze({0});
  }

  setOutTensor(ConvFlipWeightsOp::getOutIndex(), convWeights);
}

namespace {
OpxCreator<ConvOpx> convpxCreator(Onnx::Operators::Conv_1);
OpxCreator<ConvWeightsGradOpx>
    convWeightsGradOpxCreator(Onnx::GradOperators::ConvWeightsGrad);
OpxCreator<ConvFlipWeightsGradOpx>
    convFlipWeightsGradOpxCreator(Onnx::CustomOperators::ConvFlipWeights);

} // namespace

} // namespace popx
} // namespace poponnx
