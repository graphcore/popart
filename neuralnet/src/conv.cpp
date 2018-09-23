#include <neuralnet/conv.hpp>
#include <neuralnet/error.hpp>
#include <neuralnet/tensor.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <cblas.h>
#pragma clang diagnostic pop // stop ignoring warnings

namespace neuralnet {

ConvOp::ConvOp(const onnx::NodeProto &node, Graph *pgraph)
    : HasReceptiveFieldOp(node, pgraph) {
  if (input.n()) {
    throw error("Conv with bias case not handled");
  }
}

std::vector<std::unique_ptr<Op>> ConvOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new ConvDataGradOp(this)));
  upops.emplace_back(std::unique_ptr<Op>(new ConvWeightsGradOp(this)));
  return upops;
}

bool ConvOp::readyToCreateGradients(std::set<int> &s0) const {
  return s0.size() == output.n();
}

void ConvWeightsGradOp::setup() {
  output.tensor(0)->info = input.tensor(convOp->weightsInIndex())->info;
}

void ConvDataGradOp::setup() {
  output.tensor(0)->info = input.tensor(convOp->dataInIndex())->info;
}

void ConvOp::setup0() {
  nOutChans = input.tensor(1)->info.dim(0);
  // setting groups from the input tensor,
  // we could also use the value in nAtts, as
  // "group" is required property of the ONNX conv op
  group = nInChans / input.tensor(1)->info.dim(1);
}

// ConvOp attributes only MIGHT contain the kernel shape,
// but we can ALWAYS get it directly from the kernel tensor
// at input index 1 so this is the preferred way to do it
void ConvOp::setSpatial() {
  spatial.reserve(nSpatialDims);
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    spatial.push_back(input.tensor(1)->info.dim(spDim + 2));
  }
}

int64_t ConvOp::getNOutChans() const { return nOutChans; }

ConvWeightsGradOp::ConvWeightsGradOp(ConvOp *op_)
    : GradOp({"ConvWeightsGrad", op_->pgraph, {}, getNeuralNetDomain()}),
      convOp(op_) {}

Op *ConvWeightsGradOp::getNonGradOp() { return convOp; }

const std::vector<GradInOutMapper> &ConvWeightsGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo =
      createConvWeightsGradInfo();
  return inInfo;
}

std::map<int, int> ConvWeightsGradOp::createConvWeightsGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the conv ops weight input index
  return {{0, convOp->weightsInIndex()}};
}

const std::map<int, int> &ConvWeightsGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createConvWeightsGradOutToIn();
  return outInfo;
}

std::vector<GradInOutMapper>
ConvWeightsGradOp::createConvWeightsGradInfo() const {
  // input at index 0 : gradient of output of conv
  // input at index 1 : data input to conv
  return {{0, 0, GradOpInType::GRADOUT},
          {1, convOp->dataInIndex(), GradOpInType::IN}};
}

ConvDataGradOp::ConvDataGradOp(ConvOp *op_)
    : GradOp({"ConvDataGrad", op_->pgraph, {}, getNeuralNetDomain()}),
      convOp(op_) {}

Op *ConvDataGradOp::getNonGradOp() { return convOp; }

const std::vector<GradInOutMapper> &ConvDataGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createConvDataGradInfo();
  return inInfo;
}

std::map<int, int> ConvDataGradOp::createConvDataGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the conv ops input input index
  return {{0, convOp->dataInIndex()}};
}

const std::map<int, int> &ConvDataGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createConvDataGradOutToIn();
  return outInfo;
}

std::vector<GradInOutMapper> ConvDataGradOp::createConvDataGradInfo() const {
  // input at index 0 : gradient of output of conv
  // input at index 1 : weights input to conv
  return {{0, 0, GradOpInType::GRADOUT},
          {1, convOp->weightsInIndex(), GradOpInType::IN}};
}

void ConvDataGradOp::imposeTopoCons() {
  // ig we wanted to say that
  // this will is first consumer of the gradient:
  // input.tensor(0)->consumers.setTopoFirst(this);
}

} // namespace neuralnet
