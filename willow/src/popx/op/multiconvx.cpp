// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <map>
#include <memory>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <utility>
#include <vector>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <poplin/ConvParams.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MultiConvolution.hpp>
#include <popart/op/multiconv.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/multiconvx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/op/convbase.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/convbasex.hpp"

namespace popart {
class Op;

namespace popx {

MultiConvOpx::MultiConvOpx(Op *op, Devicex *devicex)
    : MultiConvBaseOpx(op, devicex) {
  verifyOp<MultiConvOp>(op, {Onnx::CustomOperators::MultiConv_1});
}

std::vector<poplin::multiconv::CreateTensorArgs>
MultiConvOpx::getCreateTensorArgs(const poplar::DebugNameAndId &dnai) const {
  auto &op = getOp<MultiConvOp>();
  std::vector<poplin::multiconv::CreateTensorArgs> allCreateTensorArgs;

  for (int i = 0; i < op.numConvs(); i++) {
    poplin::multiconv::CreateTensorArgs createTensorArgs;
    createTensorArgs.params  = getPoplarConvParams(op.getParameters(i));
    createTensorArgs.options = getConvOptions(i, getFwdPassFlagString());
    // NOTE: Would be better if poplin took a debug object.
    createTensorArgs.name = dnai.getPathName();

    allCreateTensorArgs.push_back(createTensorArgs);
  }

  return allCreateTensorArgs;
}

poplar::OptionFlags MultiConvOpx::getGlobalOptions() const {
  poplar::OptionFlags optionFlags;
  for (auto key_val :
       getOp<MultiConvOp>().getConvOptions().getGlobalOptions()) {
    optionFlags.set(key_val.first, key_val.second);
  }
  return optionFlags;
}

snap::Tensor
MultiConvOpx::createWeightsInput(const poplar::DebugNameAndId &dnai,
                                 int convIndex) const {
  return snap::Tensor{
      poplin::multiconv::createWeights(graph().getPoplarGraph(),
                                       getCreateTensorArgs(dnai),
                                       static_cast<unsigned>(convIndex),
                                       getGlobalOptions(),
                                       &dv_p->convCache),
      graph()};
}
snap::Tensor MultiConvOpx::createDataInput(const poplar::DebugNameAndId &dnai,
                                           int convIndex) const {
  return snap::Tensor{
      poplin::multiconv::createInput(graph().getPoplarGraph(),
                                     getCreateTensorArgs(dnai),
                                     static_cast<unsigned>(convIndex),
                                     getGlobalOptions(),
                                     &dv_p->convCache),
      graph()};
}

std::vector<snap::Tensor>
MultiConvOpx::convolve(snap::program::Sequence &prog,
                       const std::vector<snap::Tensor> &weights) const {
  std::vector<poplin::multiconv::ConvolutionArgs> allConvArgs;
  MultiConvBaseOp &op = getOp<MultiConvBaseOp>();

  for (int i = 0; i < op.numConvs(); i++) {
    poplin::multiconv::ConvolutionArgs convArgs;
    convArgs.inputs =
        getInTensor(MultiConvBaseOp::getDataInIndex(i)).getPoplarTensor();
    convArgs.weights = weights[i].getPoplarTensor();
    convArgs.params  = getPoplarConvParams(op.getParameters(i));
    convArgs.options = getConvOptions(i);

    allConvArgs.push_back(convArgs);
  }
  auto pOuts = poplin::multiconv::convolution(graph().getPoplarGraph(),
                                              allConvArgs,
                                              false,
                                              prog.getPoplarSequence(),
                                              debugContext("multiConvolution"),
                                              getGlobalOptions(),
                                              &dv_p->convCache);
  std::vector<snap::Tensor> outs;
  outs.reserve(pOuts.size());
  for (auto out : pOuts) {
    outs.push_back(snap::Tensor{out, graph()});
  }
  return outs;
}

MultiConvWeightsGradOpx::MultiConvWeightsGradOpx(Op *op, Devicex *devicex)
    : MultiConvWeightsGradBaseOpx(op, devicex) {
  verifyOp<MultiConvWeightsGradOpx>(
      op, {Onnx::GradOperators::MultiConvWeightsGrad});
}

std::vector<snap::Tensor> MultiConvWeightsGradOpx::calculateWeightDeltas(
    snap::program::Sequence &prog) const {

  // The multiconv api for calculating weight deltas is deprecated.
  // Use the standard convolution api
  MultiConvWeightsGradOp &op = getOp<MultiConvWeightsGradOp>();
  std::vector<snap::Tensor> outTensors;

  for (int i = 0; i < op.numConvs(); i++) {
    const snap::Tensor &zDelta = getInTensor(op.getGradConvolvedInIndex(i));
    const snap::Tensor &acts   = getInTensor(op.getPreConvolvedInIndex(i));

    snap::Tensor wGrad = snap::Tensor{
        poplin::calculateWeightDeltas(graph().getPoplarGraph(),
                                      zDelta.getPoplarTensor(),
                                      acts.getPoplarTensor(),
                                      getPoplarConvParams(op.getParameters(i)),
                                      prog.getPoplarSequence(),
                                      debugContext("weightDeltas"),
                                      getConvOptions(i),
                                      &dv_p->convCache),
        graph()};
    outTensors.push_back(wGrad);
  }
  return outTensors;
}

poplar::OptionFlags MultiConvWeightsGradOpx::getGlobalOptions() const {
  poplar::OptionFlags optionFlags;
  for (auto key_val :
       getOp<MultiConvWeightsGradOp>().getConvOptions().getGlobalOptions()) {
    optionFlags.set(key_val.first, key_val.second);
  }
  return optionFlags;
}

namespace {
OpxCreator<MultiConvOpx>
    multiconvopxCreator({Onnx::CustomOperators::MultiConv_1});
OpxCreator<MultiConvWeightsGradOpx>
    convWeightsGradOpxCreator(Onnx::GradOperators::MultiConvWeightsGrad);
} // namespace

} // namespace popx
} // namespace popart
