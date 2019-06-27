#include <memory>
#include <poponnx/graph.hpp>
#include <poponnx/op/addbias.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/patterns/convdatagrad.hpp>
#include <poponnx/patterns/patterns.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

bool ConvDataGradPattern::matches(Op *op) const {
  return (op->opid == Onnx::GradOperators::ConvDataGrad);
}

std::vector<const Tensor *> ConvDataGradPattern::touches(Op *) const {
  return {};
}

/*
static void printConvParameters(const ConvParameters &params) {
  logging::pattern::info(
      "poponnx::ConvParameters t:{} bs:{} s:{} ks:{} c:{},{} g:{} i:({} {} {} "
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

  auto flip = dynamic_cast<ConvFlipWeightsOp *>(
      makeReplacementOpInIr(Onnx::CustomOperators::ConvFlipWeights, op));
  auto conv = dynamic_cast<ConvOp *>(
      makeReplacementOpInIr(Onnx::Operators::Conv_1, op));

  // Get the data grad conv parameters
  ConvParameters bwdConvParams = convdatagrad->getParameters();

  // Get the input shape
  auto fwdConvInputShape = convdatagrad->getCloneOfCreator()->getInputShape();

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
  flip->setPhase(Phase::BWD);

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
  conv->setPhase(Phase::BWD);

  return true;
}

namespace {
static PatternCreator<ConvDataGradPattern>
    convDataGradPattern(PreAliasPatternType::CONVDATAGRAD, "ConvDataGrad");
}

} // namespace poponnx
