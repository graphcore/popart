// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/op/conv.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/convx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/tensor.hpp>

#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>

namespace popart {

namespace popx {

namespace {
void addPartialsType(const ConvPartialsType &partialsType,
                     poplar::OptionFlags &optionFlags) {
  switch (partialsType) {
  case ConvPartialsType::HALF: {
    optionFlags.set("partialsType", "half");
    break;
  }
  case ConvPartialsType::FLOAT: {
    optionFlags.set("partialsType", "float");
    break;
  }
  default:
    throw error("Bad ConvPartialsType {}", static_cast<int>(partialsType));
  }
}

void addAvailableMemoryProportion(
    boost::optional<float> availableMemoryProportion,
    poplar::OptionFlags &optionFlags) {
  if (availableMemoryProportion) {
    optionFlags.set("availableMemoryProportion",
                    std::to_string(*availableMemoryProportion));
  }
}
} // namespace

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
      popType(param.type),
      param.batchSize,
      vXtoY<int64_t, size_t>(param.inputShape),
      vXtoY<int64_t, size_t>(param.kernelShape),
      param.numInChannelsPerGroup,
      param.numOutChannelsPerGroup,
      param.numGroups,
      // InputTransform inputTransform
      {
          vXtoY<int64_t, unsigned>(param.inputTransformation.lowerTruncation),
          vXtoY<int64_t, unsigned>(param.inputTransformation.upperTruncation),
          vXtoY<int64_t, unsigned>(param.inputTransformation.dilation),
          vXtoY<int64_t, unsigned>(param.inputTransformation.lowerPadding),
          vXtoY<int64_t, unsigned>(param.inputTransformation.upperPadding),
          param.inputTransformation.flip,
      },
      // InputTransform kernelTransform
      {
          vXtoY<int64_t, unsigned>(param.kernelTransformation.lowerTruncation),
          vXtoY<int64_t, unsigned>(param.kernelTransformation.upperTruncation),
          vXtoY<int64_t, unsigned>(param.kernelTransformation.dilation),
          vXtoY<int64_t, unsigned>(param.kernelTransformation.lowerPadding),
          vXtoY<int64_t, unsigned>(param.kernelTransformation.upperPadding),
          param.kernelTransformation.flip,
      },
      // OutputTransform outputTransform
      {vXtoY<int64_t, unsigned>(param.outputTransformation.lowerTruncation),
       vXtoY<int64_t, unsigned>(param.outputTransformation.upperTruncation),
       vXtoY<int64_t, unsigned>(param.outputTransformation.stride),
       vXtoY<int64_t, unsigned>(param.outputTransformation.lowerPadding),
       vXtoY<int64_t, unsigned>(param.outputTransformation.upperPadding)});
}

static ConvParameters
convertPoplarConvParameters(const poplin::ConvParams &popParams) {

  ConvParameters params;
  params.batchSize   = popParams.batchSize;
  params.inputShape  = vXtoY<std::size_t, int64_t>(popParams.inputFieldShape);
  params.kernelShape = vXtoY<std::size_t, int64_t>(popParams.kernelShape);
  params.numInChannelsPerGroup  = popParams.getNumInputChansPerConvGroup();
  params.numOutChannelsPerGroup = popParams.getNumOutputChansPerConvGroup();
  params.numGroups              = popParams.getNumConvGroups();

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

  auto canonicalizedPopParams = popParams.canonicalize();

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

static poplar::Tensor
reshapeOnnxWeightsForPoplar(const poplar::Tensor &weights,
                            std::size_t chansOut,
                            std::size_t chansIn,
                            const ConvParameters &params) {
  std::size_t groups               = params.numGroups;
  std::vector<int64_t> kernelShape = params.kernelShape;

  std::vector<std::size_t> weightsShape{groups, chansOut, chansIn};
  for (auto i : kernelShape) {
    weightsShape.push_back(i);
  }
  return weights.reshape(weightsShape);
}

poplar::OptionFlags ConvOpx::getOptions() const {
  ConvOp &op = getOp<ConvOp>();

  // Work out the option based on the phase of the op
  // Conv can be bwd depending on the phase.
  PoplarOptions *options = nullptr;
  if (op.toLoss == PathToLoss::Yes && op.fromLoss == PathFromLoss::No) {
    options = &dv_p->fwdConvOptions;
  } else if (op.toLoss == PathToLoss::No && op.fromLoss == PathFromLoss::Yes) {
    options = &dv_p->bwdConvOptions;
  } else {
    logging::opx::warn(
        "Conv has undefined phase, defaulting to fwdConvOptions");
    options = &dv_p->fwdConvOptions;
  }

  // Maybe set the partials
  auto optionFlags = options->toOptionFlags();
  addPartialsType(op.getPartialsType(), optionFlags);
  // Maybe set the available memory proportion
  addAvailableMemoryProportion(op.getAvailableMemoryProportion(), optionFlags);
  return optionFlags;
}

void ConvOpx::grow(poplar::program::Sequence &prog) const {

  ConvOp &op          = getOp<ConvOp>();
  const auto &in      = getInTensor(ConvOp::getDataInIndex());
  const auto &weights = getInTensor(ConvOp::getWeightsInIndex());

  auto params    = op.getParameters();
  auto weights5D = reshapeOnnxWeightsForPoplar(weights,
                                               params.numOutChannelsPerGroup,
                                               params.numInChannelsPerGroup,
                                               params);

  poplin::ConvParams popConvParams = getPoplarConvParams(op.getParameters());

  auto outTensor = poplin::convolution(graph(),
                                       in,
                                       weights5D,
                                       popConvParams,
                                       false,
                                       prog,
                                       debugPrefix("convolution"),
                                       getOptions(),
                                       &(dv_p->convCache));

  // Log the report plan
  std::stringstream ss;
  poplin::reportPlanInfo(
      ss, graph(), popConvParams, getOptions(), &(dv_p->convCache));
  logging::opx::debug("Conv {} plan", op_p->str());
  logging::log(logging::Module::opx, logging::Level::Debug, ss.str());

  setOutTensor(ConvOp::getOutIndex(), outTensor);
}

void ConvWeightsGradOpx::grow(poplar::program::Sequence &prog) const {
  ConvWeightsGradOp &gradOp = getOp<ConvWeightsGradOp>();
  const ConvOp *convOp      = gradOp.getCloneOfCreator();

  const poplar::Tensor &zDelta =
      getInTensor(ConvWeightsGradOp::getGradConvolvedInIndex());
  const poplar::Tensor &activations =
      getInTensor(ConvWeightsGradOp::getPreConvolvedInIndex());

  auto optionFlags = dv_p->wuConvOptions.toOptionFlags();
  addPartialsType(convOp->getPartialsType(), optionFlags);
  addAvailableMemoryProportion(convOp->getAvailableMemoryProportion(),
                               optionFlags);

  poplar::Tensor wGrad = poplin::calculateWeightDeltas(
      graph(),
      zDelta,
      activations,
      getPoplarConvParams(convOp->getParameters()),
      prog,
      debugPrefix("weightDeltas"),
      optionFlags,
      &dv_p->convCache);

  // Log the report plan
  std::stringstream ss;
  poplin::reportWeightUpdatePlanInfo(
      ss,
      graph(),
      getPoplarConvParams(convOp->getParameters()),
      optionFlags,
      &(dv_p->convCache));
  logging::opx::debug("ConvWeightUpdate {} plan", op_p->str());
  logging::log(logging::Module::opx, logging::Level::Debug, ss.str());

  // Shape of weights Popart Tensor of forward Op
  // auto fwdShape = convOp->inInfo(convOp->getWeightsInIndex()).shape_szt(); //
  // segfault
  auto fwdShape = gradOp.outInfo(ConvWeightsGradOp::getOutIndex()).shape_szt();

  // If poplar::Tensor has an extra 0th (grouping) dimension, as in
  //   IR shape:             [   a*b, c, d, e]
  //   poplar::Tensor shape: [a, b  , c, d, e]
  // then reshape to combine grouping and outChannels dimensions, to
  // match the Ir tensor shape
  if (wGrad.rank() == 5 && fwdShape.size() == 4) {
    auto wGradShape = wGrad.shape();
    if (std::equal(
            wGradShape.begin() + 2, wGradShape.end(), fwdShape.begin() + 1) &&
        wGradShape[0] * wGradShape[1] == fwdShape[0]) {
      wGrad = wGrad.reshape(fwdShape);
    }
  }

  setOutTensor(ConvWeightsGradOp::getOutIndex(), wGrad);
}

ConvOpx::ConvOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ConvOp>(op, {Onnx::Operators::Conv_1, Onnx::Operators::Conv_11});

  ConvOp &cOp = getOp<ConvOp>();
  if (cOp.dataIn()->info.rank() != 4 || cOp.weightsIn()->info.rank() != 4) {
    throw error("Poplar only supports convolutions with 2 spatial dimensions");
  }
}

bool ConvOpx::createsEquiv(int ind0, const Opx *opx1, int ind1) const {
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

  const ConvOpx *rhs = dynamic_cast<const ConvOpx *>(opx1);
  auto &rhsOp        = rhs->getOp<ConvOp>();
  if (lhsOp.getParameters() != rhsOp.getParameters()) {
    return false;
  }

  return true;
}

InputCreatorType ConvOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanCreate;
}

poplar::Tensor ConvOpx::createInput(InIndex index,
                                    const std::string &name) const {

  auto &op = getOp<ConvOp>();

  auto optionFlags = dv_p->fwdConvOptions.toOptionFlags();
  addPartialsType(op.getPartialsType(), optionFlags);
  addAvailableMemoryProportion(op.getAvailableMemoryProportion(), optionFlags);

  if (index == ConvOp::getWeightsInIndex()) {
    poplar::Tensor input =
        poplin::createWeights(graph(),                                 // graph
                              getPoplarConvParams(op.getParameters()), // params
                              name,                                    // name
                              optionFlags,     // options
                              &dv_p->convCache // cache
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
      // Check that the shapes are compatible.
      // They should be related as shown below:
      //   IR shape :            [        a,        b, c, d]
      //   poplar::Tensor shape: [groups, a/groups, b, c, d]
      auto groups  = op.getParameters().numGroups;
      auto ptshape = input.shape();
      auto irshape = op_p->inInfo(index).shape_szt();

      if (std::equal(ptshape.begin() + 2, ptshape.end(), irshape.begin() + 1) &&
          ptshape[0] == groups && ptshape[1] * groups == irshape[0]) {
        input = input.reshape(irshape);
      }
    }
    return input;
  } else if (index == ConvOp::getDataInIndex()) {
    return poplin::createInput(
        graph(),                                 // graph
        getPoplarConvParams(op.getParameters()), // params
        name,                                    // name
        optionFlags,                             // options
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

  auto &op    = getOp<ConvFlipWeightsOp>();
  auto params = op.getParameters();

  poplar::Tensor weights = getInTensor(ConvFlipWeightsOp::getInIndex());
  // swap In Out channels
  auto weights5D = reshapeOnnxWeightsForPoplar(weights,
                                               params.numInChannelsPerGroup,
                                               params.numOutChannelsPerGroup,
                                               params);

  auto fwdOptions            = dv_p->bwdConvOptions;
  fwdOptions.options["pass"] = "TRAINING_FWD";

  auto optionFlags = fwdOptions.toOptionFlags();
  addPartialsType(op.getPartialsType(), optionFlags);
  addAvailableMemoryProportion(op.getAvailableMemoryProportion(), optionFlags);

  poplin::ConvParams popConvParams = getPoplarConvParams(params);

  auto convWeights =
      poplin::createWeights(graph(),
                            popConvParams,
                            inTensor(ConvFlipWeightsOp::getInIndex())->str() +
                                sNameDelimiter + "flipped",
                            optionFlags,
                            &dv_p->convCache);

  // weightsTransposeChansFlipXY must be called on each group individually
  for (int i = 0; i < params.numGroups; i++) {
    // dim 0 of weights5D and convWeights are the groups.
    // slice off group i from weights5D and convWeights.
    auto w = weights5D.slice(i, i + 1, 0);
    auto c = convWeights.slice(i, i + 1, 0);

    // call weightsTransposeChansFlipXY on group i of weights5D and convWeights.
    poplin::weightsTransposeChansFlipXY(
        graph(),
        w,
        c,
        seq,
        debugPrefix(logging::format("group{}_transposeXY", i)));
  }

  auto newShape = convWeights.shape();
  newShape[2]   = newShape[2] * newShape[0];
  newShape[0]   = 1;
  convWeights   = convWeights.reshape(newShape);

  // Taken the 1 off the front convWeights if it was added.
  if (weights.rank() != weights5D.rank()) {
    convWeights = convWeights.squeeze({0});
  }

  setOutTensor(ConvFlipWeightsOp::getOutIndex(), convWeights);
}

namespace {
OpxCreator<ConvOpx> convpxCreator({Onnx::Operators::Conv_1,
                                   Onnx::Operators::Conv_11});
OpxCreator<ConvWeightsGradOpx>
    convWeightsGradOpxCreator(Onnx::GradOperators::ConvWeightsGrad);
OpxCreator<ConvFlipWeightsGradOpx>
    convFlipWeightsGradOpxCreator(Onnx::CustomOperators::ConvFlipWeights);

} // namespace

} // namespace popx
} // namespace popart
