// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "popart/ir.hpp"
#include "popart/util/float8util.hpp"
#include <map>
#include <memory>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <utility>
#include <vector>
#include <poplar/MetadataCreation.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Quarter.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poplin/Convolution.hpp>
#include <popart/op/conv.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/convx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

#include "popart/datatype.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op/convbase.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/convbasex.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class Op;

namespace popx {

ConvOpx::ConvOpx(Op *op, Devicex *devicex) : MultiConvBaseOpx(op, devicex) {
  verifyOp<ConvOp>(op, {Onnx::Operators::Conv_1, Onnx::Operators::Conv_11});
}

snap::Tensor ConvOpx::createWeightsInput(const poplar::DebugNameAndId &dnai,
                                         int convIndex) const {
  poplar::Tensor weightsInput = poplin::createWeights(
      graph().getPoplarGraph(),
      getPoplarConvParams(getOp<ConvOp>().getParameters()),
      dnai,
      getConvOptions(convIndex, getFwdPassFlagString()),
      &dv_p->convCache);

  if (weightsInput.elementType() == poplar::QUARTER) {
    weightsInput = weightsInput.reinterpret(poplar::UNSIGNED_CHAR);
  }
  return snap::Tensor{weightsInput, graph()};
}

snap::Tensor ConvOpx::createDataInput(const poplar::DebugNameAndId &dnai,
                                      int convIndex) const {
  poplar::Tensor dataInput =
      poplin::createInput(graph().getPoplarGraph(),
                          getPoplarConvParams(getOp<ConvOp>().getParameters()),
                          dnai,
                          getConvOptions(convIndex, getFwdPassFlagString()),
                          &dv_p->convCache);

  if (dataInput.elementType() == poplar::QUARTER) {
    dataInput = dataInput.reinterpret(poplar::UNSIGNED_CHAR);
  }
  return snap::Tensor{dataInput, graph()};
}

InputCreatorType ConvOpx::getInputCreatorType(InIndex idx) const {
  if (idx == ConvOp::getLog2ScaleInIndex()) {
    return InputCreatorType::Deadend;
  }
  return InputCreatorType::CanCreate;
}

std::vector<snap::Tensor>
ConvOpx::convolve(snap::program::Sequence &prog,
                  const std::vector<snap::Tensor> &weights) const {
  ConvOp &op = getOp<ConvOp>();

  poplar::Tensor data = getInTensor(ConvOp::getDataInIndex()).getPoplarTensor();
  poplar::Tensor convWeights = weights[0].getPoplarTensor();

  if (op.isPow2ScaledConv()) {
    poplar::Tensor log2Scale =
        getInTensor(ConvOp::getLog2ScaleInIndex()).getPoplarTensor();

    if (op.getIr().getSessionOptions().throwIfLog2ScaleTensorNotInRange) {
      auto assertProg = createAssertLog2ScaleInRangeProg(
          graph().getPoplarGraph(), log2Scale, -32, 32);

      prog.getPoplarSequence().add(assertProg);
    }

    data = reinterpretCastUInt8ToQuarter(
        graph(),
        data,
        toPoplarQuarterFormat(op.inInfo(ConvOp::getDataInIndex()).dataType()),
        log2Scale,
        prog,
        debugContext());
    convWeights = reinterpretCastUInt8ToQuarter(
        graph(),
        convWeights,
        toPoplarQuarterFormat(
            op.inInfo(ConvOp::getWeightsInIndex()).dataType()),
        0,
        prog,
        debugContext());
  }

  auto outTensor =
      snap::Tensor{poplin::convolution(graph().getPoplarGraph(),
                                       data,
                                       convWeights,
                                       getPoplarConvParams(op.getParameters()),
                                       false,
                                       prog.getPoplarSequence(),
                                       debugContext("convolution"),
                                       getConvOptions(0),
                                       &(dv_p->convCache)),
                   graph()};
  return {outTensor};
}

ConvWeightsGradOpx::ConvWeightsGradOpx(Op *op, Devicex *devicex)
    : MultiConvWeightsGradBaseOpx(op, devicex) {
  verifyOp<ConvWeightsGradOp>(op, Onnx::GradOperators::ConvWeightsGrad);
}

std::vector<snap::Tensor>
ConvWeightsGradOpx::calculateWeightDeltas(snap::program::Sequence &prog) const {
  ConvWeightsGradOp &gradOp = getOp<ConvWeightsGradOp>();

  const snap::Tensor &zDelta = getInTensor(gradOp.getGradConvolvedInIndex());
  const snap::Tensor &acts   = getInTensor(gradOp.getPreConvolvedInIndex());

  snap::Tensor wGrad = snap::Tensor{
      poplin::calculateWeightDeltas(graph().getPoplarGraph(),
                                    zDelta.getPoplarTensor(),
                                    acts.getPoplarTensor(),
                                    getPoplarConvParams(gradOp.getParameters()),
                                    prog.getPoplarSequence(),
                                    debugContext("weightDeltas"),
                                    getConvOptions(),
                                    &dv_p->convCache),
      graph()};
  return {wGrad};
}

ConvFlipWeightsGradOpx::ConvFlipWeightsGradOpx(Op *op_, Devicex *devicex_)
    : PopOpx(op_, devicex_) {
  verifyOp<ConvFlipWeightsOp>(op_, Onnx::CustomOperators::ConvFlipWeights);
}

void ConvFlipWeightsGradOpx::grow(snap::program::Sequence &seq) const {

  auto &op    = getOp<ConvFlipWeightsOp>();
  auto params = op.getParameters();

  snap::Tensor weights = getInTensor(ConvFlipWeightsOp::getInIndex());
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
      graph().getPoplarGraph(),
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
        graph().getPoplarGraph(),
        w.getPoplarTensor(),
        c,
        seq.getPoplarSequence(),
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
      snap::Tensor{
          convWeights.reshape(
              outTensor(ConvFlipWeightsOp::getOutIndex())->info.shape_szt()),
          graph()});
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
