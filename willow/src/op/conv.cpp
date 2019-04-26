#include <algorithm>
#include <memory>
#include <vector>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

// These are utility functions that are need by the Ir Conv.
namespace popx {
ConvParameters getConvGradParameters(const ConvParameters &fwdParams);
ConvParameters canonicalizeConvParams(const ConvParameters &param);
} // namespace popx

ConvOp::ConvOp(const OperatorIdentifier &_opid,
               const HasReceptiveFieldOp::Settings &settings_)
    : HasReceptiveFieldOp(_opid, settings_) {}

const Tensor *ConvOp::dataIn() const { return inTensor(getDataInIndex()); }

const Tensor *ConvOp::weightsIn() const {
  return inTensor(getWeightsInIndex());
}

std::vector<std::unique_ptr<Op>> ConvOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ConvDataGradOp>(*this));
  upops.emplace_back(make_unique<ConvWeightsGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> ConvOp::clone() const { return make_unique<ConvOp>(*this); }

void ConvWeightsGradOp::setup() { outInfo(getOutIndex()) = weightsInfo; }

void ConvDataGradOp::setup() { outInfo(getOutIndex()) = dataInfo; }

Shape ConvOp::getOutShape() const {
  if (!outputShape.empty()) {
    return outputShape;
  } else {
    return HasReceptiveFieldOp::getOutShape();
  }
}

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

void ConvOp::setup() {

  // Call the base class
  HasReceptiveFieldOp::setup();

  // record the inputShape so we can use it later for the bwd pass
  inputShape = inShape(ConvOp::getDataInIndex());

  // Setup the conv parameters
  params.type           = outType;
  params.batchSize      = batchSize;
  params.inputShape     = spatialD;
  params.kernelShape    = spatialK;
  params.numInChannels  = nInChans;
  params.numOutChannels = getNOutChans();
  params.numGroups      = group;

  std::vector<int64_t> zeros(nSpatialDims, 0);
  std::vector<int64_t> ones(nSpatialDims, 1);
  std::vector<bool> falses(nSpatialDims, false);

  params.inputTransformation.lowerTruncation = zeros;
  params.inputTransformation.upperTruncation = zeros;
  params.inputTransformation.dilation        = ones;
  params.inputTransformation.lowerPadding    = lowerPads();
  params.inputTransformation.upperPadding    = upperPads();
  params.inputTransformation.flip            = falses;

  params.kernelTransformation.lowerTruncation = zeros;
  params.kernelTransformation.upperTruncation = zeros;
  params.kernelTransformation.dilation        = dilations;
  params.kernelTransformation.lowerPadding    = zeros;
  params.kernelTransformation.upperPadding    = zeros;
  params.kernelTransformation.flip            = falses;

  params.outputTransformation.lowerTruncation = zeros;
  params.outputTransformation.upperTruncation = zeros;
  params.outputTransformation.stride          = strides;
  params.outputTransformation.lowerPadding    = zeros;
  params.outputTransformation.upperPadding    = zeros;
}

const ConvOp *ConvWeightsGradOp::getCloneOfCreator() const {
  return dynamic_cast<const ConvOp *>(cloneOfCreator.get());
}

const ConvOp *ConvDataGradOp::getCloneOfCreator() const {
  return dynamic_cast<const ConvOp *>(cloneOfCreator.get());
}

int64_t ConvOp::getNOutChans() const { return nOutChans; }

static void appendConvParameterAttributes(const ConvParameters &params,
                                          OpSerialiserBase &os) {

  // The original conv caching  canonicalize the parameter that went into the
  // cache key
  ConvParameters p = popx::canonicalizeConvParams(params);

  os.appendAttribute("__batchsize", p.batchSize);
  os.appendAttribute("__batchsize", p.numInChannels);
  os.appendAttribute("__numOutChannels", p.numOutChannels);
  os.appendAttribute("__inputShape", p.inputShape);
  os.appendAttribute("__kernelShape", p.kernelShape);

  os.appendAttribute("__input.lowerTruncation",
                     p.inputTransformation.lowerTruncation);
  os.appendAttribute("__input.upperTruncation",
                     p.inputTransformation.lowerTruncation);
  os.appendAttribute("__input.dilation", p.inputTransformation.dilation);
  os.appendAttribute("__input.lowerPadding",
                     p.inputTransformation.lowerPadding);
  os.appendAttribute("__input.upperPadding",
                     p.inputTransformation.upperPadding);
  os.appendAttribute("__input.flip",
                     vBooltoY<int64_t>(p.inputTransformation.flip));

  os.appendAttribute("__kernel.lowerTruncation",
                     p.kernelTransformation.lowerTruncation);
  os.appendAttribute("__kernel.upperTruncation",
                     p.kernelTransformation.lowerTruncation);
  os.appendAttribute("__kernel.dilation", p.kernelTransformation.dilation);
  os.appendAttribute("__kernel.lowerPadding",
                     p.kernelTransformation.lowerPadding);
  os.appendAttribute("__kernel.upperPadding",
                     p.kernelTransformation.upperPadding);
  os.appendAttribute("__kernel.flip",
                     vBooltoY<int64_t>(p.kernelTransformation.flip));

  os.appendAttribute("__output.lowerTruncation",
                     p.outputTransformation.lowerTruncation);
  os.appendAttribute("__output.upperTruncation",
                     p.outputTransformation.lowerTruncation);
  os.appendAttribute("__output.stride", p.outputTransformation.stride);
  os.appendAttribute("__output.lowerPadding",
                     p.outputTransformation.lowerPadding);
  os.appendAttribute("__output.upperPadding",
                     p.outputTransformation.upperPadding);
}

void ConvOp::appendAttributes(OpSerialiserBase &os) const {
  HasReceptiveFieldOp::appendAttributes(os);
  appendConvParameterAttributes(params, os);
}

ConvWeightsGradOp::ConvWeightsGradOp(const ConvOp &op_)
    : Op(Onnx::GradOperators::ConvWeightsGrad, op_.getSettings()),
      cloneOfCreator(op_.clone()),
      weightsInfo(op_.inInfo(ConvOp::getWeightsInIndex())) {
  // we want this Op to be executed early, so that the weight
  // update can be performed as early as possible, thus making
  // weight gradient tensors non-live. TODO : same for matmul
  priority = std::numeric_limits<double>::max();
}

void ConvWeightsGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendForwardOp(getCloneOfCreator());
  appendConvParameterAttributes(getCloneOfCreator()->getParameters(), os);
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

ConvDataGradOp::ConvDataGradOp(const ConvOp &op_)
    : Op(Onnx::GradOperators::ConvDataGrad, op_.getSettings()),
      cloneOfCreator(op_.clone()),
      dataInfo(op_.inInfo(ConvOp::getDataInIndex())) {

  params = popx::getConvGradParameters(getCloneOfCreator()->getParameters());
}

void ConvDataGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendForwardOp(getCloneOfCreator());
  appendConvParameterAttributes(params, os);
}

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

ConvFlipWeightsOp::ConvFlipWeightsOp(const OperatorIdentifier &opid_,
                                     const Op::Settings &settings_)
    : Op(opid_, settings_) {

  // This op is used exclusively in the backwards pass as an input to the
  // ConvDataGradOp. Since it acts only on an input to the graph, it has no
  // dependencies. Demote its priority to ensure this is scheduled after
  // all forwards pass operations (where liveness is typically greatess).

  // Priorty should be more negative for ops further into the backwards pass
  // (i.e. not all the same value) to ensure all ConvFlipWeightsOps don't
  // execute right at the start of the backwards pass.

  priority = -id;
}

ConvFlipWeightsOp::~ConvFlipWeightsOp() {}

void ConvFlipWeightsOp::setup() {

  auto &weightsIn = inInfo(getInIndex());

  // Switch the first two dimensions
  Shape weightsOutShape;
  weightsOutShape.push_back(weightsIn.dim(1));
  weightsOutShape.push_back(weightsIn.dim(0));
  for (int i = 2; i < weightsIn.shape().size(); ++i) {
    weightsOutShape.push_back(weightsIn.dim(i));
  }

  outInfo(getOutIndex()) = {weightsIn.dataType(), weightsOutShape};
}

namespace {
static OpCreator<ConvOp> convOpCreator(
    Onnx::Operators::Conv_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      HasReceptiveFieldOp::Settings receptiveSettings(
          settings.graph, settings.name, settings.scope);
      receptiveSettings.setFromAttributes(attr);

      return std::unique_ptr<Op>(new ConvOp(_opid, receptiveSettings));
    },
    true);

static OpCreator<ConvFlipWeightsOp>
    convFlipWeightsOpCreator(Onnx::CustomOperators::ConvFlipWeights);
} // namespace

} // namespace poponnx
