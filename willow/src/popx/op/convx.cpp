// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/op/conv.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/convx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/tensor.hpp>

namespace popart {

namespace popx {

ConvOpx::ConvOpx(Op *op, Devicex *devicex) : MultiConvBaseOpx(op, devicex) {
  verifyOp<ConvOp>(op, {Onnx::Operators::Conv_1, Onnx::Operators::Conv_11});
}

poplar::Tensor ConvOpx::createWeightsInput(const poplar::DebugNameAndId &dnai,
                                           int convIndex) const {
  return poplin::createWeights(
      graph(),
      getPoplarConvParams(getOp<ConvOp>().getParameters()),
      dnai,
      getConvOptions(convIndex, getFwdPassFlagString()),
      &dv_p->convCache);
}
poplar::Tensor ConvOpx::createDataInput(const poplar::DebugNameAndId &dnai,
                                        int convIndex) const {
  return poplin::createInput(
      graph(),
      getPoplarConvParams(getOp<ConvOp>().getParameters()),
      dnai,
      getConvOptions(convIndex, getFwdPassFlagString()),
      &dv_p->convCache);
}

std::vector<poplar::Tensor>
ConvOpx::convolve(poplar::program::Sequence &prog,
                  const std::vector<poplar::Tensor> &weights) const {
  ConvOp &op     = getOp<ConvOp>();
  auto outTensor = poplin::convolution(graph(),
                                       getInTensor(ConvOp::getDataInIndex()),
                                       weights[0],
                                       getPoplarConvParams(op.getParameters()),
                                       false,
                                       prog,
                                       debugContext("convolution"),
                                       getConvOptions(0),
                                       &(dv_p->convCache));
  return {outTensor};
}

ConvWeightsGradOpx::ConvWeightsGradOpx(Op *op, Devicex *devicex)
    : MultiConvWeightsGradBaseOpx(op, devicex) {
  verifyOp<ConvWeightsGradOp>(op, Onnx::GradOperators::ConvWeightsGrad);
}

std::vector<poplar::Tensor> ConvWeightsGradOpx::calculateWeightDeltas(
    poplar::program::Sequence &prog) const {
  ConvWeightsGradOp &gradOp = getOp<ConvWeightsGradOp>();

  const poplar::Tensor &zDelta = getInTensor(gradOp.getGradConvolvedInIndex());
  const poplar::Tensor &acts   = getInTensor(gradOp.getPreConvolvedInIndex());

  poplar::Tensor wGrad =
      poplin::calculateWeightDeltas(graph(),
                                    zDelta,
                                    acts,
                                    getPoplarConvParams(gradOp.getParameters()),
                                    prog,
                                    debugContext("weightDeltas"),
                                    getConvOptions(),
                                    &dv_p->convCache);
  return {wGrad};
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

  poplar::OptionFlags optionFlags;
  for (auto key_val : getOp<ConvFlipWeightsOp>().getConvOptions()) {
    optionFlags.set(key_val.first, key_val.second);
  }
  optionFlags.set("pass", "TRAINING_FWD");

  auto convWeights = poplin::createWeights(
      graph(),
      getPoplarConvParams(params),
      debugContext(inTensor(ConvFlipWeightsOp::getInIndex())->str() +
                   sNameDelimiter + "flipped"),
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
        debugContext(logging::format("group{}_transposeXY", i)));
  }

  auto newShape = convWeights.shape();
  newShape[2]   = newShape[2] * newShape[0];
  newShape[0]   = 1;
  convWeights   = convWeights.reshape(newShape);

  // Taken the 1 off the front convWeights if it was added.
  if (weights.rank() != weights5D.rank()) {
    convWeights = convWeights.squeeze({0});
  }

  setOutTensor(
      ConvFlipWeightsOp::getOutIndex(),
      convWeights.reshape(
          outTensor(ConvFlipWeightsOp::getOutIndex())->info.shape_szt()));
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
