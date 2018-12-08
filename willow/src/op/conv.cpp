#include <algorithm>
#include <memory>
#include <vector>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ConvOp::ConvOp(const onnx::NodeProto &node, Ir *_pir)
    : HasReceptiveFieldOp(node, _pir) {}

const Tensor *ConvOp::dataIn() const { return inTensor(getDataInIndex()); }

const Tensor *ConvOp::weightsIn() const {
  return inTensor(getWeightsInIndex());
}

std::vector<std::unique_ptr<Op>> ConvOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ConvDataGradOp>(this));
  upops.emplace_back(make_unique<ConvWeightsGradOp>(this));
  return upops;
}

std::unique_ptr<Op> ConvOp::clone() const { return make_unique<ConvOp>(*this); }

void ConvWeightsGradOp::setup() { outInfo(getOutIndex()) = weightsInfo; }

void ConvDataGradOp::setup() { outInfo(getOutIndex()) = dataInfo; }

void ConvOp::setup0() {
  nOutChans = weightsIn()->info.dim(0);
  // setting groups from the input tensor,
  // we could also use the value in nAtts, as
  // "group" is required property of the ONNX conv op
  group = nInChans / weightsIn()->info.dim(1);

  // Get the attribute wether we should do graph caching for this convolution.
  nAtts.setIfPresent(cacheOperation, "__cache_operation");
  // Override if caching has been disabled for the whole graph.
  const auto &sessionOptions = pir->getSessionOptions();
  cacheOperation &= sessionOptions.enableConvolutionGraphCaching;
}

// ConvOp attributes only MIGHT contain the kernel shape,
// but we can ALWAYS get it directly from the kernel tensor
// at input index 1 so this is the preferred way to do it
void ConvOp::setSpatialK() {
  spatialK.resize(nSpatialDims);
  spatialK.reserve(nSpatialDims);
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    spatialK[spDim] = weightsIn()->info.dim(spDim + 2);
  }
}

const ConvOp *ConvWeightsGradOp::getCloneOfCreator() const {
  return dynamic_cast<const ConvOp *>(cloneOfCreator.get());
}

const ConvOp *ConvDataGradOp::getCloneOfCreator() const {
  return dynamic_cast<const ConvOp *>(cloneOfCreator.get());
}

int64_t ConvOp::getNOutChans() const { return nOutChans; }

ConvWeightsGradOp::ConvWeightsGradOp(ConvOp *op_)
    : Op({OpType::CONVWEIGHTSGRAD, op_->pir, {}}), cloneOfCreator(op_->clone()),
      weightsInfo(op_->inInfo(ConvOp::getWeightsInIndex())) {
  // we want this Op to be executed early, so that the weight
  // update can be performed as early as possible, thus making
  // weight gradient tensors non-live. TODO : same for matmul
  priority = std::numeric_limits<double>::max();
}

const std::vector<GradInOutMapper> &ConvWeightsGradOp::gradInputInfo() const {
  // input at index getGradConvolvedIn() (0) : gradient of output of conv
  // input at index getPreConvolvedIn() (1)  : data input to conv
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradConvolvedInIndex(), ConvOp::getOutIndex(), GradOpInType::GRADOUT},
      {getPreConvolvedInIndex(), ConvOp::getDataInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &ConvWeightsGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the conv ops weight input index
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ConvOp::getWeightsInIndex()}};
  return outInfo;
}

ConvDataGradOp::ConvDataGradOp(ConvOp *op_)
    : Op({OpType::CONVDATAGRAD, op_->pir, {}}), cloneOfCreator(op_->clone()),
      dataInfo(op_->inInfo(ConvOp::getDataInIndex())) {}

const std::vector<GradInOutMapper> &ConvDataGradOp::gradInputInfo() const {
  // input at index getGradConvolvedIn() : gradient of output of conv
  // input at index getWeightsIn()       : weights input to conv
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradConvolvedInIndex(), ConvOp::getOutIndex(), GradOpInType::GRADOUT},
      {getWeightsInIndex(), ConvOp::getWeightsInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &ConvDataGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the conv ops input input index
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ConvOp::getDataInIndex()}};
  return outInfo;
}

} // namespace poponnx
