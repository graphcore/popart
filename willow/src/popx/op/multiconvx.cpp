// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <popart/op/multiconv.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/multiconvx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/poplaroptionsx.hpp>

namespace popart {

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

poplar::Tensor
MultiConvOpx::createWeightsInput(const poplar::DebugNameAndId &dnai,
                                 int convIndex) const {
  return poplin::multiconv::createWeights(graph(),
                                          getCreateTensorArgs(dnai),
                                          static_cast<unsigned>(convIndex),
                                          getGlobalOptions(),
                                          &dv_p->convCache);
}
poplar::Tensor MultiConvOpx::createDataInput(const poplar::DebugNameAndId &dnai,
                                             int convIndex) const {
  return poplin::multiconv::createInput(graph(),
                                        getCreateTensorArgs(dnai),
                                        static_cast<unsigned>(convIndex),
                                        getGlobalOptions(),
                                        &dv_p->convCache);
}

std::vector<poplar::Tensor>
MultiConvOpx::convolve(poplar::program::Sequence &prog,
                       const std::vector<poplar::Tensor> &weights) const {
  std::vector<poplin::multiconv::ConvolutionArgs> allConvArgs;
  MultiConvBaseOp &op = getOp<MultiConvBaseOp>();

  for (int i = 0; i < op.numConvs(); i++) {
    poplin::multiconv::ConvolutionArgs convArgs;
    convArgs.inputs  = getInTensor(MultiConvBaseOp::getDataInIndex(i));
    convArgs.weights = weights[i];
    convArgs.params  = getPoplarConvParams(op.getParameters(i));
    convArgs.options = getConvOptions(i);

    allConvArgs.push_back(convArgs);
  }
  return poplin::multiconv::convolution(graph(),
                                        allConvArgs,
                                        false,
                                        prog,
                                        debugContext("multiConvolution"),
                                        getGlobalOptions(),
                                        &dv_p->convCache);
}

MultiConvWeightsGradOpx::MultiConvWeightsGradOpx(Op *op, Devicex *devicex)
    : MultiConvWeightsGradBaseOpx(op, devicex) {
  verifyOp<MultiConvWeightsGradOpx>(
      op, {Onnx::GradOperators::MultiConvWeightsGrad});
}

std::vector<poplar::Tensor> MultiConvWeightsGradOpx::calculateWeightDeltas(
    poplar::program::Sequence &prog) const {

  // The multiconv api for calculating weight deltas is deprecated.
  // Use the standard convolution api
  MultiConvWeightsGradOp &op = getOp<MultiConvWeightsGradOp>();
  std::vector<poplar::Tensor> outTensors;

  for (int i = 0; i < op.numConvs(); i++) {
    const poplar::Tensor &zDelta = getInTensor(op.getGradConvolvedInIndex(i));
    const poplar::Tensor &acts   = getInTensor(op.getPreConvolvedInIndex(i));

    poplar::Tensor wGrad =
        poplin::calculateWeightDeltas(graph(),
                                      zDelta,
                                      acts,
                                      getPoplarConvParams(op.getParameters(i)),
                                      prog,
                                      debugContext("weightDeltas"),
                                      getConvOptions(i),
                                      &dv_p->convCache);
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
