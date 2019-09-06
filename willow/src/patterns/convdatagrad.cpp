#include <memory>
#include <popart/graph.hpp>
#include <popart/op/addbias.hpp>
#include <popart/op/conv.hpp>
#include <popart/patterns/convdatagrad.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>

namespace popart {

bool ConvDataGradPattern::matches(Op *op) const {
  return (op->opid == Onnx::GradOperators::ConvDataGrad);
}

std::vector<const Tensor *> ConvDataGradPattern::touches(Op *) const {
  return {};
}

/*
static void printConvParameters(const ConvParameters &params) {
  logging::pattern::info(
      "popart::ConvParameters t:{} bs:{} s:{} ks:{} c:{},{} g:{} i:({} {} {} "
      "{} {} {}) k:({} {} {} {} {} {}) o:({} {} {} {} {})",
      static_cast<int>(params.type),
      params.batchSize,
      params.inputShape,
      params.kernelShape,
      params.numInChannels,
      params.numOutChannels,
      params.numGroups,
      params.inputTransformation.lowerTruncation,
      params.inputTransformation.upperTruncation,
      params.inputTransformation.dilation,
      params.inputTransformation.lowerPadding,
      params.inputTransformation.upperPadding,
      vXtoY<bool, int>(params.inputTransformation.flip),
      params.kernelTransformation.lowerTruncation,
      params.kernelTransformation.upperTruncation,
      params.kernelTransformation.dilation,
      params.kernelTransformation.lowerPadding,
      params.kernelTransformation.upperPadding,
      vXtoY<bool, int>(params.kernelTransformation.flip),
      params.outputTransformation.lowerTruncation,
      params.outputTransformation.upperTruncation,
      params.outputTransformation.stride,
      params.outputTransformation.lowerPadding,
      params.outputTransformation.upperPadding);
}
*/

bool ConvDataGradPattern::apply(Op *op) const {

  auto weights_in     = op->inTensor(ConvDataGradOp::getWeightsInIndex());
  auto gradConvIn_out = op->inTensor(ConvDataGradOp::getGradConvolvedInIndex());
  auto grad_out       = op->outTensor(ConvDataGradOp::getOutIndex());

  const auto convdatagrad = dynamic_cast<ConvDataGradOp *>(op);
  const auto fwdOp        = convdatagrad->getCloneOfCreator();

  auto flip = dynamic_cast<ConvFlipWeightsOp *>(
      makeReplacementOpInIr(Onnx::CustomOperators::ConvFlipWeights, op));
  auto conv = dynamic_cast<ConvOp *>(
      makeReplacementOpInIr(Onnx::Operators::Conv_1, op));

  // Inherit the partials type from the forward op
  flip->setPartialsType(fwdOp->getPartialsType());
  conv->setPartialsType(fwdOp->getPartialsType());

  // Inherit the availableMemoryProportion from the forward op
  flip->setAvailableMemoryProportion(fwdOp->getAvailableMemoryProportion());
  conv->setAvailableMemoryProportion(fwdOp->getAvailableMemoryProportion());

  // Get the data grad conv parameters
  ConvParameters bwdConvParams = convdatagrad->getParameters();

  // Get the input shape
  auto fwdConvInputShape = fwdOp->getInputShape();

  // Remove the ConvGradOp
  convdatagrad->disconnectAllInputs();
  convdatagrad->disconnectAllOutputs();
  convdatagrad->getGraph().eraseOp(convdatagrad->id);

  // Configure the flip weight op
  flip->connectInTensor(ConvFlipWeightsOp::getInIndex(), weights_in->id);
  flip->createAndConnectOutTensor(ConvFlipWeightsOp::getOutIndex(),
                                  createIntermediateTensorId(weights_in->id));

  flip->setup();
  flip->setParameters(bwdConvParams);

  // Configure the conv op for the bwd pass
  conv->connectInTensor(ConvOp::getWeightsInIndex(),
                        flip->outTensor(ConvFlipWeightsOp::getOutIndex())->id);
  conv->connectInTensor(ConvOp::getDataInIndex(), gradConvIn_out->id);
  conv->connectOutTensor(ConvOp::getOutIndex(), grad_out->id);

  // - set the onnx attributes based on the bwd conv parameters
  std::copy(bwdConvParams.inputTransformation.lowerPadding.begin(),
            bwdConvParams.inputTransformation.lowerPadding.end(),
            std::back_inserter(conv->pads));
  std::copy(bwdConvParams.inputTransformation.upperPadding.begin(),
            bwdConvParams.inputTransformation.upperPadding.end(),
            std::back_inserter(conv->pads));
  conv->strides   = bwdConvParams.kernelTransformation.dilation;
  conv->dilations = bwdConvParams.outputTransformation.stride;

  conv->setOutputShape(fwdConvInputShape);
  conv->setup();
  conv->setParameters(bwdConvParams);

  return true;
}

namespace {
static PatternCreator<ConvDataGradPattern>
    convDataGradPattern(PreAliasPatternType::CONVDATAGRAD, "ConvDataGrad");
}

} // namespace popart
