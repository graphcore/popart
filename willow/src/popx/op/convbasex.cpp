// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/op/convbase.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/convbasex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/poplaroptionsx.hpp>

#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>

namespace popart {

namespace popx {

std::set<TensorId> MultiConvBaseOpx::mustExistBeforeCreate(InIndex) const {
  // creation of both weights and of input are done
  // without requiring the pre-existance of any
  // other poplar::Tensor, so returning empty TensorId vector
  return {};
}

InputCreatorType MultiConvBaseOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanCreate;
}

bool MultiConvBaseOpx::createsEquiv(int ind0, const Opx *opx1, int ind1) const {
  // if opx1 is not a MultiConvBaseOpx, it does not create the same
  // poplar::Tensor
  if (opx1->op_p->opid != op_p->opid) {
    return false;
  }

  // if opx1 (which we now know is MultiConvBaseOpx) would create the tensor at
  // a different input index, creation is not equivalent
  if (ind0 != ind1) {
    return false;
  }

  // finally, check that the convolution parameters are the same
  auto &lhsOp                 = getOp<MultiConvBaseOp>();
  const MultiConvBaseOpx *rhs = dynamic_cast<const MultiConvBaseOpx *>(opx1);
  auto &rhsOp                 = rhs->getOp<MultiConvBaseOp>();

  // get conv indices from input indices
  int lhsConvIndex = MultiConvBaseOp::getConvIndexFromInIndex(ind0);
  int rhsConvIndex = MultiConvBaseOp::getConvIndexFromInIndex(ind1);

  if (lhsOp.getParameters(lhsConvIndex) != rhsOp.getParameters(rhsConvIndex)) {
    return false;
  }

  return true;
}

bool MultiConvBaseOpx::isWeightsInIndex(InIndex index) const {
  auto &op = getOp<MultiConvBaseOp>();
  for (int i = 0; i < op.numConvs(); i++) {
    if (index == MultiConvBaseOp::getWeightsInIndex(i)) {
      return true;
    }
  }
  return false;
}

bool MultiConvBaseOpx::isDataInIndex(InIndex index) const {
  auto &op = getOp<MultiConvBaseOp>();
  for (int i = 0; i < op.numConvs(); i++) {
    if (index == MultiConvBaseOp::getDataInIndex(i)) {
      return true;
    }
  }
  return false;
}

void MultiConvBaseOpx::verifyCacheSizeUnchanged(size_t beforeCacheSize) const {
  if (beforeCacheSize != dv_p->convCache.size()) {
    // TODO: T34143 replace this with error
    logging::opx::info(
        "Pre-planning failed for {}. Its plan was not found in the cache",
        op_p->str());
  }
}

poplar::Tensor
MultiConvBaseOpx::createInput(InIndex index,
                              const poplar::DebugNameAndId &dnai) const {
  auto &op       = getOp<MultiConvBaseOp>();
  auto convIndex = MultiConvBaseOp::getConvIndexFromInIndex(index);

  if (isWeightsInIndex(index)) {
    poplar::Tensor input = createWeightsInput(dnai, convIndex);

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
    if (input.rank() == op_p->inRank(index) + 1) {
      // Check that the shapes are compatible.
      // They should be related as shown below:
      //   IR shape :            [        a,        b, c, d]
      //   poplar::Tensor shape: [groups, a/groups, b, c, d]
      auto groups  = op.getParameters(convIndex).numGroups;
      auto ptshape = input.shape();
      auto irshape = op_p->inInfo(index).shape_szt();

      if (std::equal(ptshape.begin() + 2, ptshape.end(), irshape.begin() + 1) &&
          ptshape[0] == groups && ptshape[1] * groups == irshape[0]) {
        input = input.reshape(irshape);
      }
    }
    return input;
  } else if (isDataInIndex(index)) {
    return createDataInput(dnai, convIndex);
  } else {
    throw error("conv opx cannot create tensor at this index yet");
  }
}

std::string MultiConvBaseOpx::getFwdPassFlagString() const {
  if (op_p->getIr().getExecutionMode() == Ir::ExecutionMode::Training) {
    return "TRAINING_FWD";
  } else {
    return "INFERENCE_FWD";
  }
}

poplar::OptionFlags MultiConvBaseOpx::getConvOptions(int convIndex,
                                                     std::string pass) const {
  poplar::OptionFlags optionFlags;
  for (auto key_val :
       getOp<MultiConvBaseOp>().getConvOptions().getConvOptions(convIndex)) {
    optionFlags.set(key_val.first, key_val.second);
  }

  // If 'pass' is unspecified, decide based on position of Op in the graph,
  // relative to the loss
  if (pass.empty()) {
    // default to 'fwd pass' flag setting
    pass = getFwdPassFlagString();
    if (op_p->toLoss == PathToLoss::No && op_p->fromLoss == PathFromLoss::Yes) {
      pass = "TRAINING_BWD";
    }
  }
  optionFlags.set("pass", pass);

  return optionFlags;
}

void MultiConvBaseOpx::grow(poplar::program::Sequence &prog) const {
  MultiConvBaseOp &op = getOp<MultiConvBaseOp>();
  std::vector<poplar::Tensor> allWeights;

  auto cacheSize = dv_p->convCache.size();

  for (int i = 0; i < op.numConvs(); i++) {
    auto params    = op.getParameters(i);
    auto weights   = getInTensor(MultiConvBaseOp::getWeightsInIndex(i));
    auto weights5D = reshapeOnnxWeightsForPoplar(weights,
                                                 params.numOutChannelsPerGroup,
                                                 params.numInChannelsPerGroup,
                                                 params);
    allWeights.push_back(weights5D);

    // Log the report plan
    std::stringstream ss;
    poplin::reportPlanInfo(ss,
                           graph(),
                           getPoplarConvParams(params),
                           getConvOptions(i),
                           &(dv_p->convCache));
    logging::opx::debug("Conv {} plan ({})", op_p->str(), i);
    logging::log(logging::Module::opx, logging::Level::Debug, ss.str());
  }

  std::vector<poplar::Tensor> outTensors = convolve(prog, allWeights);

  verifyCacheSizeUnchanged(cacheSize);

  for (int i = 0; i < op.numConvs(); i++) {
    setOutTensor(MultiConvBaseOp::getOutIndex(i), outTensors[i]);
  }
}

poplar::OptionFlags
MultiConvWeightsGradBaseOpx::getConvOptions(int convIndex) const {
  poplar::OptionFlags optionFlags;
  for (auto key_val :
       getOp<MultiConvWeightsGradBaseOp>().getConvOptions().getConvOptions(
           convIndex)) {
    optionFlags.set(key_val.first, key_val.second);
  }
  optionFlags.set("pass", "TRAINING_WU");
  return optionFlags;
}

void MultiConvWeightsGradBaseOpx::verifyCacheSizeUnchanged(
    size_t beforeCacheSize) const {
  if (beforeCacheSize != dv_p->convCache.size()) {
    throw internal_error(
        "Pre-planning failed for {}. Its plan was not found in the cache",
        op_p->str());
  }
}

void MultiConvWeightsGradBaseOpx::grow(poplar::program::Sequence &prog) const {
  MultiConvWeightsGradBaseOp &op = getOp<MultiConvWeightsGradBaseOp>();

  auto cacheSize                         = dv_p->convCache.size();
  std::vector<poplar::Tensor> outTensors = calculateWeightDeltas(prog);
  verifyCacheSizeUnchanged(cacheSize);

  for (int i = 0; i < op.numConvs(); i++) {
    // If poplar::Tensor has an extra 0th (grouping) dimension, as in
    //   IR shape:             [   a*b, c, d, e]
    //   poplar::Tensor shape: [a, b  , c, d, e]
    // then reshape to combine grouping and outChannels dimensions, to
    // match the Ir tensor shape
    auto fwdShape =
        op.outInfo(MultiConvWeightsGradBaseOp::getOutIndex(i)).shape_szt();
    if (outTensors[i].rank() == fwdShape.size() + 1) {
      auto wGradShape = outTensors[i].shape();
      if (std::equal(
              wGradShape.begin() + 2, wGradShape.end(), fwdShape.begin() + 1) &&
          wGradShape[0] * wGradShape[1] == fwdShape[0]) {
        outTensors[i] = outTensors[i].reshape(fwdShape);
      }
    }
    setOutTensor(MultiConvWeightsGradBaseOp::getOutIndex(i), outTensors[i]);
  }
}

ConvParameters getConvGradParameters(const ConvParameters &fwdParams) {
  poplin::ConvParams popBwdParams =
      poplin::getGradientParams(getPoplarConvParams(fwdParams));

  ConvParameters bwdParams = convertPoplarConvParameters(popBwdParams);
  bwdParams.type           = fwdParams.type;

  return bwdParams;
}

ConvParameters getConvWeightUpdateParameters(const ConvParameters &fwdParams) {
  poplin::ConvParams popBwdParams =
      poplin::getWeightUpdateParams(getPoplarConvParams(fwdParams));

  ConvParameters bwdParams = convertPoplarConvParameters(popBwdParams);
  bwdParams.type           = fwdParams.type;

  return bwdParams;
}

ConvParameters canonicalizeConvParams(const ConvParameters &param) {
  poplin::ConvParams popParams = getPoplarConvParams(param);

  auto canonicalizedPopParams = popParams.canonicalize();

  ConvParameters result = convertPoplarConvParameters(canonicalizedPopParams);
  result.type           = param.type;
  return result;
}

poplar::Tensor reshapeOnnxWeightsForPoplar(const poplar::Tensor &weights,
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

poplin::ConvParams getPoplarConvParams(const ConvParameters &param) {
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

ConvParameters
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

} // namespace popx
} // namespace popart
