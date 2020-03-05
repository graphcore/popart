#include <algorithm>
#include <memory>
#include <vector>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/conv.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::string toString(const ConvPartialsType &x) {
  switch (x) {
  case ConvPartialsType::HALF:
    return "ConvPartialsType::HALF";
  case ConvPartialsType::FLOAT:
    return "ConvPartialsType::FLOAT";
  default:
    throw error("Bad ConvPartialsType '{}'", static_cast<int>(x));
  }
}

std::ostream &operator<<(std::ostream &os, const ConvPartialsType &x) {
  os << toString(x);
  return os;
}

// These are utility functions that are need by the Ir Conv.
namespace popx {
ConvParameters getConvGradParameters(const ConvParameters &fwdParams);
ConvParameters canonicalizeConvParams(const ConvParameters &param);
} // namespace popx

ConvOp::ConvOp(const OperatorIdentifier &_opid,
               int64_t group_,
               const ConvPartialsType &partialsType_,
               boost::optional<float> availableMemoryProportion_,
               const HasReceptiveFieldOp::Settings &settings_)
    : HasReceptiveFieldOp(_opid, settings_), group(group_),
      partialsType(partialsType_),
      availableMemoryProportion(availableMemoryProportion_) {}

const Tensor *ConvOp::dataIn() const { return inTensor(getDataInIndex()); }

const Tensor *ConvOp::weightsIn() const {
  return inTensor(getWeightsInIndex());
}

std::vector<std::unique_ptr<Op>> ConvOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<ConvDataGradOp>(*this));
  upops.emplace_back(std::make_unique<ConvWeightsGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> ConvOp::clone() const {
  return std::make_unique<ConvOp>(*this);
}

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

  if (group == 0) {
    throw error("group attribute in {} must be greater than zero", debugName());
  }

  if (nInChans != weightsIn()->info.dim(1) * group) {
    throw error(
        "Invalid value for group ({}) in {}. number of input channels ({}) / "
        "group ({}) should be equal to the weight inputs second dimension ({})",
        group,
        debugName(),
        nInChans,
        group,
        weightsIn()->info.dim(1));
  }
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
  params.type                   = outType;
  params.batchSize              = batchSize;
  params.inputShape             = spatialD;
  params.kernelShape            = spatialK;
  params.numInChannelsPerGroup  = nInChans / group;
  params.numOutChannelsPerGroup = getNOutChans() / group;
  params.numGroups              = group;

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
  os.appendAttribute("__numInChannelsPerGroup", p.numInChannelsPerGroup);
  os.appendAttribute("__numOutChannelsPerGroup", p.numOutChannelsPerGroup);
  os.appendAttribute("__inputShape", p.inputShape);
  os.appendAttribute("__kernelShape", p.kernelShape);
  os.appendAttribute("__groups", p.numGroups);

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

void ConvOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  HasReceptiveFieldOp::appendOutlineAttributes(os);
  os.appendAttribute("partialsType", toString(partialsType));
  if (availableMemoryProportion) {
    os.appendAttribute("availableMemoryProportion", *availableMemoryProportion);
  }
  appendConvParameterAttributes(params, os);
}

ConvWeightsGradOp::ConvWeightsGradOp(const ConvOp &op_)
    : Op(Onnx::GradOperators::ConvWeightsGrad, op_.getSettings()),
      cloneOfCreator(op_.clone()),
      weightsInfo(op_.inInfo(ConvOp::getWeightsInIndex())) {}

std::unique_ptr<Op> ConvWeightsGradOp::clone() const {
  return std::make_unique<ConvWeightsGradOp>(*this);
}

void ConvWeightsGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
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

std::unique_ptr<Op> ConvDataGradOp::clone() const {
  return std::make_unique<ConvDataGradOp>(*this);
}

void ConvDataGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
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
    : Op(opid_, settings_) {}

std::unique_ptr<Op> ConvFlipWeightsOp::clone() const {
  return std::make_unique<ConvFlipWeightsOp>(*this);
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

void ConvFlipWeightsOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("partialsType", toString(partialsType));
  if (availableMemoryProportion) {
    os.appendAttribute("availableMemoryProportion", *availableMemoryProportion);
  }
}

namespace {
ConvPartialsType fromString(const std::string &s) {
  if (s == "HALF" || s == "half") {
    return ConvPartialsType::HALF;
  } else if (s == "FLOAT" || s == "float") {
    return ConvPartialsType::FLOAT;
  } else {
    throw error("Unable to get ConvPartialsType from string '{}'", s);
  }
}

static OpDefinition convOpDef(
    {OpDefinition::Inputs({
         {"X", {{DataType::FLOAT, DataType::FLOAT16}}},
         {"W", {{DataType::FLOAT, DataType::FLOAT16}}},
         {"B", {{DataType::FLOAT, DataType::FLOAT16}}},
     }),
     OpDefinition::Outputs({{"Y", {{DataType::FLOAT, DataType::FLOAT16}}}}),
     OpDefinition::Attributes({
         {"auto_pad", {"NOTSET"}}, // don't support. auto pad does not seem
         // deprecated from conv
         {"dilations", {"*"}},
         {"group", {"*"}},
         {"kernel_shape", {"*"}}, // Do we support this?
         {"pads", {"*"}},
         {"strides", {"*"}},
     })});

static OpCreator<ConvOp> convOpCreator(
    OpDefinitions({
        {Onnx::Operators::Conv_1, convOpDef},
        {Onnx::Operators::Conv_11, convOpDef},
    }),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      HasReceptiveFieldOp::Settings receptiveSettings(
          settings.graph, settings.name, settings.scope);
      receptiveSettings.setFromAttributes(attr);

      int64_t group = attr.getAttribute<Attributes::Int>("group", 1);

      auto partialsType = ConvPartialsType::FLOAT;
      boost::optional<float> availableMemoryProportion = boost::none;

      // try set the partials from an attribute
      if (attr.hasAttribute(sPartialsTypeAttribute)) {
        std::string partialsTypeAttr =
            attr.getAttribute<Attributes::String>(sPartialsTypeAttribute);
        partialsType = fromString(partialsTypeAttr);
      }
      // otherwise see if partialsType was set in the convolution options
      else {
        auto &opts = settings.getIr().getSessionOptions().convolutionOptions;
        auto partialsTypeOpt = opts.find("partialsType");
        if (partialsTypeOpt != opts.end()) {
          partialsType = fromString(partialsTypeOpt->second);
        }
      }

      // try set the availMemAttribute from an attribute
      if (attr.hasAttribute(sAvailMemAttribute)) {
        availableMemoryProportion =
            attr.getAttribute<Attributes::Float>(sAvailMemAttribute);
      }

      return std::unique_ptr<Op>(new ConvOp(_opid,
                                            group,
                                            partialsType,
                                            availableMemoryProportion,
                                            receptiveSettings));
    },
    true);

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    convFlipWeightsOpDef({OpDefinition::Inputs({{"input", T}}),
                          OpDefinition::Outputs({{"output", T}}),
                          OpDefinition::Attributes({})});

static OpCreator<ConvFlipWeightsOp> convFlipWeightsOpCreator(OpDefinitions({
    {Onnx::CustomOperators::ConvFlipWeights, convFlipWeightsOpDef},
}));
} // namespace

} // namespace popart
