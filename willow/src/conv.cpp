#include <memory>
#include <poponnx/conv.hpp>
#include <poponnx/error.hpp>
#include <poponnx/tensor.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
// The CPU backend may require cblas.h
// #include <cblas.h>
#pragma clang diagnostic pop // stop ignoring warnings

namespace willow {

int convDataInIndex() { return 0; }
int convWeightsInIndex() { return 1; }

ConvOp::ConvOp(const onnx::NodeProto &node, Ir *pir)
    : HasReceptiveFieldOp(node, pir) {
  if (input.n()) {
    throw error("Conv with bias case not handled");
  }
}

const Tensor *ConvOp::dataIn() const { return input.tensor(convDataInIndex()); }

const Tensor *ConvOp::weightsIn() const {
  return input.tensor(convWeightsInIndex());
}

std::vector<std::unique_ptr<Op>> ConvOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new ConvDataGradOp(this)));
  upops.emplace_back(std::unique_ptr<Op>(new ConvWeightsGradOp(this)));
  return upops;
}

std::unique_ptr<Op> ConvOp::clone() const {
  return std::unique_ptr<Op>(new ConvOp(*this));
}

void ConvWeightsGradOp::setup() { output.tensor(0)->info = weightsInfo; }

void ConvDataGradOp::setup() { output.tensor(0)->info = dataInfo; }

void ConvOp::setup0() {
  nOutChans = weightsIn()->info.dim(0);
  // setting groups from the input tensor,
  // we could also use the value in nAtts, as
  // "group" is required property of the ONNX conv op
  group = nInChans / weightsIn()->info.dim(1);
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
    : Op({"ConvWeightsGrad", op_->pir, {}, getWillowDomain()}),
      cloneOfCreator(op_->clone()),
      weightsInfo(op_->input.tensor(convWeightsInIndex())->info) {}

const std::vector<GradInOutMapper> &ConvWeightsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo =
      createConvWeightsGradInfo();
  return inInfo;
}

std::map<int, int> ConvWeightsGradOp::createConvWeightsGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the conv ops weight input index
  return {{0, convWeightsInIndex()}};
}

const std::map<int, int> &ConvWeightsGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createConvWeightsGradOutToIn();
  return outInfo;
}

int ConvWeightsGradOp::getGradConvolvedIn() const { return 0; }

int ConvWeightsGradOp::getPreConvolvedIn() const { return 1; }

std::vector<GradInOutMapper>
ConvWeightsGradOp::createConvWeightsGradInfo() const {
  // input at index getGradConvolvedIn() (0) : gradient of output of conv
  // input at index getPreConvolvedIn() (1)  : data input to conv
  return {{getGradConvolvedIn(), 0, GradOpInType::GRADOUT},
          {getPreConvolvedIn(), convDataInIndex(), GradOpInType::IN}};
}

ConvDataGradOp::ConvDataGradOp(ConvOp *op_)
    : Op({"ConvDataGrad", op_->pir, {}, getWillowDomain()}),
      cloneOfCreator(op_->clone()),
      dataInfo(op_->input.tensor(convDataInIndex())->info) {}

const std::vector<GradInOutMapper> &ConvDataGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createConvDataGradInfo();
  return inInfo;
}

std::map<int, int> ConvDataGradOp::createConvDataGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the conv ops input input index
  return {{0, convDataInIndex()}};
}

const std::map<int, int> &ConvDataGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createConvDataGradOutToIn();
  return outInfo;
}

int ConvDataGradOp::getWeightsIn() const { return 0; }

int ConvDataGradOp::getGradConvolvedIn() const { return 1; }

std::vector<GradInOutMapper> ConvDataGradOp::createConvDataGradInfo() const {
  // input at index getGradConvolvedIn() : gradient of output of conv
  // input at index getWeightsIn()       : weights input to conv
  return {{getGradConvolvedIn(), 0, GradOpInType::GRADOUT},
          {getWeightsIn(), convWeightsInIndex(), GradOpInType::IN}};
}

} // namespace willow
