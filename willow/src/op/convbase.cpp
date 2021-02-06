// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poprithms/util/stringutil.hpp>
#include <popart/ir.hpp>
#include <popart/op/convbase.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

// These are utility functions that are need by the Ir Conv.
namespace popx {
ConvParameters getConvGradParameters(const ConvParameters &fwdParams);
ConvParameters canonicalizeConvParams(const ConvParameters &param);
} // namespace popx

MultiConvOptions::MultiConvOptions(
    const std::map<std::string, std::string> sessionConvOptions,
    const Attributes &attr) {
  int64_t numConvs = attr.getAttribute<Attributes::Int>("numConvs", 1);

  // Per-conv options:
  // 1. Partials type
  if (attr.hasAttribute(sPartialsTypeAttribute)) {
    // Set from locally set attribute
    if (attr.getAttribute<Attributes::Strings>(sPartialsTypeAttribute).size()) {
      // > 1 convolution
      partialsTypes =
          attr.getAttribute<Attributes::Strings>(sPartialsTypeAttribute);
    } else {
      // only one convolution
      partialsTypes = {
          attr.getAttribute<Attributes::String>(sPartialsTypeAttribute)};
    }
  } else {
    // Set from session-wide convolution settings
    auto partialsTypeOpt = sessionConvOptions.find("partialsType");
    if (partialsTypeOpt != sessionConvOptions.end()) {
      std::vector<std::string> pts(numConvs, partialsTypeOpt->second);
      partialsTypes = pts;
    }
  }
  // Catch bad string early (i.e. before handing off to poplar)
  if (partialsTypes.size()) {
    // Convert to lower case
    for (int i = 0; i < partialsTypes.size(); i++) {
      partialsTypes[i] = poprithms::util::lowercase(partialsTypes[i]);
    }

    std::vector<std::string> supportedTypes = {"float", "half"};
    for (auto pt : partialsTypes) {
      bool isSupported =
          std::find(supportedTypes.begin(), supportedTypes.end(), pt) !=
          supportedTypes.end();
      if (!isSupported) {
        throw error("Bad partialsType '{}'", pt);
      }
    }
  }

  // 2. Available memory proportion
  if (attr.hasAttribute(sAvailMemAttribute)) {
    // Set from locally set attribute
    if (attr.getAttribute<Attributes::Floats>(sAvailMemAttribute).size()) {
      // > 1 convolution
      availableMemoryProportions =
          attr.getAttribute<Attributes::Floats>(sAvailMemAttribute);
    } else {
      // only one convolution
      availableMemoryProportions = {
          attr.getAttribute<Attributes::Float>(sAvailMemAttribute)};
    }
  } else {
    // Set from session-wide convolution settings
    auto availMemOpt = sessionConvOptions.find("availableMemoryProportion");
    if (availMemOpt != sessionConvOptions.end()) {
      std::vector<float> aMems(numConvs, std::stof(availMemOpt->second));
      availableMemoryProportions = aMems;
    }
  }
  // Catch bad setting early (i.e. before handing off to poplar)
  if (availableMemoryProportions.size()) {
    for (auto aMem : availableMemoryProportions) {
      if (aMem > 1.0f || aMem <= 0.0f) {
        throw error("availableMemoryProportion must be in (0,1]");
      }
    }
  }

  // Global options
  if (attr.hasAttribute("planType")) {
    planType = attr.getAttribute<Attributes::String>("planType");
  }
  if (attr.hasAttribute("perConvReservedTiles")) {
    perConvReservedTiles =
        attr.getAttribute<Attributes::Int>("perConvReservedTiles");
  }
  if (attr.hasAttribute("cycleBackOff")) {
    cycleBackOff = attr.getAttribute<Attributes::Float>("cycleBackOff");
  }
}

std::map<std::string, std::string>
MultiConvOptions::getConvOptions(int convIndex) const {
  std::map<std::string, std::string> strings;
  if (partialsTypes.size()) {
    strings["partialsType"] = partialsTypes[convIndex];
  }
  if (availableMemoryProportions.size()) {
    strings["availableMemoryProportion"] =
        std::to_string(availableMemoryProportions[convIndex]);
  }
  return strings;
}

std::map<std::string, std::string> MultiConvOptions::getGlobalOptions() const {
  std::map<std::string, std::string> strings;
  if (planType) {
    strings["planType"] = *(planType);
  }
  if (perConvReservedTiles) {
    strings["perConvReservedTiles"] = std::to_string(*(perConvReservedTiles));
  }
  if (cycleBackOff) {
    strings["cycleBackOff"] = std::to_string(*(cycleBackOff));
  }
  return strings;
}

MultiConvBaseOp::MultiConvBaseOp(const OperatorIdentifier &_opid,
                                 const Op::Settings &settings_,
                                 std::vector<int64_t> flatStrides_,
                                 std::vector<int64_t> flatPads_,
                                 std::vector<int64_t> flatDilations_,
                                 const AutoPad &padType_,
                                 const MultiConvOptions &convOpts_)
    : Op(_opid, settings_), flatStrides(flatStrides_), flatPads(flatPads_),
      flatDilations(flatDilations_), convOpts(convOpts_), padType(padType_) {}

Shape MultiConvBaseOp::getOutShape(int convIndex) const {
  Shape outShape(2 + getNSpatialDims(convIndex), 0);
  outShape[0] = inInfo(getDataInIndex(convIndex)).dim(0); // batch size
  outShape[1] = getNOutChans(convIndex);

  Shape spatialOutShape =
      HasReceptiveFieldOp::getSpatialOutShape(getSpatialD(convIndex),
                                              getSpatialK(convIndex),
                                              pads[convIndex],
                                              outPads[convIndex],
                                              strides[convIndex],
                                              dilations[convIndex],
                                              inDilations[convIndex],
                                              padType);

  for (int spDim = 0; spDim < getNSpatialDims(convIndex); ++spDim) {
    outShape[spDim + 2] = spatialOutShape[spDim];
  }
  return outShape;
}

void MultiConvBaseOp::setParamsFromDataGradOp(const Op *op) {
  // Set output shape and parameters
  auto dataGradOp = dynamic_cast<const MultiConvDataGradBaseOp *>(op);
  for (int i = 0; i < numConvs(); i++) {
    params.push_back(dataGradOp->getParameters(i));
  }
  restoreAttributesFromParams();
}

void MultiConvBaseOp::restoreAttributesFromParams() {
  // Restore Op parameters from Conv parameters so that setup() works
  flatStrides.clear();
  flatDilations.clear();
  flatPads.clear();
  flatOutPads.clear();

  for (auto param : params) {
    flatStrides.insert(flatStrides.end(),
                       param.outputTransformation.stride.begin(),
                       param.outputTransformation.stride.end());
    flatDilations.insert(flatDilations.end(),
                         param.kernelTransformation.dilation.begin(),
                         param.kernelTransformation.dilation.end());
    flatPads.insert(flatPads.end(),
                    param.inputTransformation.lowerPadding.begin(),
                    param.inputTransformation.lowerPadding.end());
    flatPads.insert(flatPads.end(),
                    param.inputTransformation.upperPadding.begin(),
                    param.inputTransformation.upperPadding.end());
    flatInDilations.insert(flatInDilations.begin(),
                           param.inputTransformation.dilation.begin(),
                           param.inputTransformation.dilation.end());
    flatOutPads.insert(flatOutPads.begin(),
                       param.outputTransformation.lowerPadding.begin(),
                       param.outputTransformation.lowerPadding.end());
    flatOutPads.insert(flatOutPads.begin(),
                       param.outputTransformation.upperPadding.begin(),
                       param.outputTransformation.upperPadding.end());
  }

  // The padding has been adjusted, so we can unset the AutoPad type
  padType = AutoPad::NOTSET;
}

void MultiConvBaseOp::appendConvParameterAttributes(
    const ConvParameters &params_,
    const std::string &suffix,
    OpSerialiserBase &os) {

  auto sfx = [&suffix](std::string attrName) { return attrName + suffix; };

  // The original conv caching  canonicalize the parameter that went into the
  // cache key
  ConvParameters p = popx::canonicalizeConvParams(params_);

  os.appendAttribute(sfx("__batchsize"), p.batchSize);
  os.appendAttribute(sfx("__numInChannelsPerGroup"), p.numInChannelsPerGroup);
  os.appendAttribute(sfx("__numOutChannelsPerGroup"), p.numOutChannelsPerGroup);
  os.appendAttribute(sfx("__inputShape"), p.inputShape);
  os.appendAttribute(sfx("__kernelShape"), p.kernelShape);
  os.appendAttribute(sfx("__groups"), p.numGroups);

  os.appendAttribute(sfx("__input.lowerTruncation"),
                     p.inputTransformation.lowerTruncation);
  os.appendAttribute(sfx("__input.upperTruncation"),
                     p.inputTransformation.lowerTruncation);
  os.appendAttribute(sfx("__input.dilation"), p.inputTransformation.dilation);
  os.appendAttribute(sfx("__input.lowerPadding"),
                     p.inputTransformation.lowerPadding);
  os.appendAttribute(sfx("__input.upperPadding"),
                     p.inputTransformation.upperPadding);
  os.appendAttribute(sfx("__input.flip"),
                     vBooltoY<int64_t>(p.inputTransformation.flip));

  os.appendAttribute(sfx("__kernel.lowerTruncation"),
                     p.kernelTransformation.lowerTruncation);
  os.appendAttribute(sfx("__kernel.upperTruncation"),
                     p.kernelTransformation.lowerTruncation);
  os.appendAttribute(sfx("__kernel.dilation"), p.kernelTransformation.dilation);
  os.appendAttribute(sfx("__kernel.lowerPadding"),
                     p.kernelTransformation.lowerPadding);
  os.appendAttribute(sfx("__kernel.upperPadding"),
                     p.kernelTransformation.upperPadding);
  os.appendAttribute(sfx("__kernel.flip"),
                     vBooltoY<int64_t>(p.kernelTransformation.flip));

  os.appendAttribute(sfx("__output.lowerTruncation"),
                     p.outputTransformation.lowerTruncation);
  os.appendAttribute(sfx("__output.upperTruncation"),
                     p.outputTransformation.lowerTruncation);
  os.appendAttribute(sfx("__output.stride"), p.outputTransformation.stride);
  os.appendAttribute(sfx("__output.lowerPadding"),
                     p.outputTransformation.lowerPadding);
  os.appendAttribute(sfx("__output.upperPadding"),
                     p.outputTransformation.upperPadding);
}

void MultiConvBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);

  for (int64_t i = 0; i < numConvs(); i++) {
    std::string suffix = "[" + std::to_string(i) + "]";

    appendConvParameterAttributes(params[i], suffix, os);

    // Append per-conv options
    for (auto key_val : getConvOptions().getConvOptions(i)) {
      os.appendAttribute(key_val.first + suffix, key_val.second);
    }
  }
}

void MultiConvBaseOp::setup() {
  strides.resize(numConvs());
  pads.resize(numConvs());
  outPads.resize(numConvs());
  dilations.resize(numConvs());
  inDilations.resize(numConvs());
  params.resize(numConvs());

  // Check that all parameters are either empty, or unpackable (i.e. they
  // contain N * sum(nSpatialDims) elements each, where N == 1 for dilantions
  // and strides, and N == 2 for pads.
  int64_t totalNumSpatialDims = 0;
  for (int i = 0; i < numConvs(); i++) {
    totalNumSpatialDims += getNSpatialDims(i);
  }
  if (!flatStrides.empty()) {
    if (flatStrides.size() != totalNumSpatialDims) {
      throw error(
          "Unexpected number of stride parameters for convolution op '{}'",
          str());
    }
  }
  if (!flatPads.empty()) {
    if (flatPads.size() != totalNumSpatialDims * 2) {
      throw error("Unexpected number of pad parameters for convolution op '{}'",
                  str());
    }
  }
  if (!flatOutPads.empty()) {
    if (flatOutPads.size() != totalNumSpatialDims * 2) {
      throw error("Unexpected number of pad parameters for convolution op '{}'",
                  str());
    }
  }
  if (!flatDilations.empty()) {
    if (flatDilations.size() != totalNumSpatialDims) {
      throw error(
          "Unexpected number of dilation parameters for convolution op '{}'",
          str());
    }
  }
  if (!flatInDilations.empty()) {
    if (flatInDilations.size() != totalNumSpatialDims) {
      throw error(
          "Unexpected number of dilation parameters for convolution op '{}'",
          str());
    }
  }

  // Set defaults if empty:
  // dilations - 1 along each spatial axis.
  // pads - 0 along each input axis
  // strides - 1 along each spatial axis
  // Otherwise, unpack.

  // Keep track of sizes of each conv of multiconv, for unpacking
  int64_t cumulativeSpatialDims = 0;

  for (int i = 0; i < numConvs(); i++) {
    const auto nSpatialDims = getNSpatialDims(i);

    if (flatStrides.empty()) {
      strides[i].resize(nSpatialDims, 1);
    } else {
      strides[i] = {flatStrides.begin() + cumulativeSpatialDims,
                    flatStrides.begin() + cumulativeSpatialDims + nSpatialDims};
    }
    if (flatPads.empty()) {
      pads[i].resize(2 * nSpatialDims, 0);
    } else {
      const auto cumulativePads = cumulativeSpatialDims * 2;
      pads[i]                   = {flatPads.begin() + cumulativePads,
                 flatPads.begin() + cumulativePads + (nSpatialDims * 2)};
    }
    if (flatOutPads.empty()) {
      outPads[i].resize(2 * nSpatialDims, 0);
    } else {
      const auto cumulativePads = cumulativeSpatialDims * 2;
      outPads[i]                = {flatOutPads.begin() + cumulativePads,
                    flatOutPads.begin() + cumulativePads + (nSpatialDims * 2)};
    }
    if (flatDilations.empty()) {
      dilations[i].resize(nSpatialDims, 1);
    } else {
      dilations[i] = {flatDilations.begin() + cumulativeSpatialDims,
                      flatDilations.begin() + cumulativeSpatialDims +
                          nSpatialDims};
    }
    if (flatInDilations.empty()) {
      inDilations[i].resize(nSpatialDims, 1);
    } else {
      inDilations[i] = {flatInDilations.begin() + cumulativeSpatialDims,
                        flatInDilations.begin() + cumulativeSpatialDims +
                            nSpatialDims};
    }

    cumulativeSpatialDims += nSpatialDims;
  }

  // Alter pads if necessary, based on AutoPad type
  for (int i = 0; i < numConvs(); i++) {
    const auto nSpatialDims = getNSpatialDims(i);
    const auto outShape     = getOutShape(i);
    Shape spatialO(nSpatialDims, 0);
    for (int j = 0; j < nSpatialDims; ++j) {
      spatialO[j] = outShape[j + 2];
    }
    if (padType == AutoPad::SAME_LOWER || padType == AutoPad::SAME_UPPER) {
      HasReceptiveFieldOp::alterPads(
          pads[i], spatialO, getSpatialD(i), getSpatialK(i), strides[i]);
    }
  }

  // Set up the conv parameters
  for (int i = 0; i < numConvs(); i++) {
    const auto nSpatialDims = getNSpatialDims(i);
    std::vector<int64_t> zeros(nSpatialDims, 0);
    std::vector<int64_t> ones(nSpatialDims, 1);
    std::vector<bool> falses(nSpatialDims, false);

    params[i].type        = inInfo(getDataInIndex(i)).dataType();
    params[i].batchSize   = inInfo(getDataInIndex(i)).dim(0);
    params[i].inputShape  = getSpatialD(i);
    params[i].kernelShape = getSpatialK(i);
    params[i].numInChannelsPerGroup =
        inInfo(getDataInIndex(i)).dim(1) / getGroups(i);
    params[i].numOutChannelsPerGroup = getNOutChans(i) / getGroups(i);
    params[i].numGroups              = getGroups(i);

    params[i].inputTransformation.lowerTruncation = zeros;
    params[i].inputTransformation.upperTruncation = zeros;
    params[i].inputTransformation.dilation        = inDilations[i];
    params[i].inputTransformation.lowerPadding    = lowerPads(i);
    params[i].inputTransformation.upperPadding    = upperPads(i);
    params[i].inputTransformation.flip            = falses;

    params[i].kernelTransformation.lowerTruncation = zeros;
    params[i].kernelTransformation.upperTruncation = zeros;
    params[i].kernelTransformation.dilation        = dilations[i];
    params[i].kernelTransformation.lowerPadding    = zeros;
    params[i].kernelTransformation.upperPadding    = zeros;
    params[i].kernelTransformation.flip            = falses;

    params[i].outputTransformation.lowerTruncation = zeros;
    params[i].outputTransformation.upperTruncation = zeros;
    params[i].outputTransformation.stride          = strides[i];
    params[i].outputTransformation.lowerPadding    = lowerOutPads(i);
    params[i].outputTransformation.upperPadding    = upperOutPads(i);
  }

  // Set output shapes
  for (int i = 0; i < numConvs(); i++) {
    outInfo(getOutIndex(i))
        .set(inInfo(getDataInIndex(i)).dataType(), getOutShape(i));
  }
}

MultiConvWeightsGradBaseOp::MultiConvWeightsGradBaseOp(
    const MultiConvBaseOp &op_,
    const OperatorIdentifier &opid_)
    : Op(opid_, op_.settings), convOpts(op_.getConvOptions()) {

  for (int i = 0; i < op_.numConvs(); i++) {
    // Set returns of gradInputInfo and gradOutToNonGradIn

    // input at index getGradConvolvedIn(i) : gradient of output of conv
    // input at index getPreConvolvedIn(i)  : data input to conv
    inInfo.push_back({getGradConvolvedInIndex(i),
                      MultiConvBaseOp::getOutIndex(i),
                      GradOpInType::GradOut});
    inInfo.push_back({getPreConvolvedInIndex(i),
                      MultiConvBaseOp::getDataInIndex(i),
                      GradOpInType::In});

    // the grad-op output at index i corresponds
    // to the conv ops weight input index i
    gradOutInfo.emplace(getOutIndex(i), MultiConvBaseOp::getWeightsInIndex(i));

    // Set the output TensorInfo
    weightsInfo.push_back(op_.inInfo(MultiConvBaseOp::getWeightsInIndex(i)));

    // Set the params for each convolution
    params.push_back(op_.getParameters(i));
  }

  if (getIr().getSessionOptions().executionPhaseSettings.phases < 2 &&
      getIr().getSessionOptions().batchSerializationSettings.factor < 2) {
    settings.schedulePriority = std::numeric_limits<double>::lowest();
  }
}

void MultiConvWeightsGradBaseOp::setup() {
  for (int i = 0; i < weightsInfo.size(); i++) {
    outInfo(getOutIndex(i)) = weightsInfo.at(i);
  }
}

void MultiConvWeightsGradBaseOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);

  for (int64_t i = 0; i < numConvs(); i++) {
    std::string suffix = "[" + std::to_string(i) + "]";

    MultiConvBaseOp::appendConvParameterAttributes(params[i], suffix, os);

    // Append per-conv options
    for (auto key_val : getConvOptions().getConvOptions(i)) {
      os.appendAttribute(key_val.first + suffix, key_val.second);
    }
  }
}

MultiConvDataGradBaseOp::MultiConvDataGradBaseOp(
    const MultiConvBaseOp &op_,
    const OperatorIdentifier &opid_)
    : Op(opid_, op_.settings), convOpts(op_.getConvOptions()) {
  // Set returns of gradInputInfo and gradOutToNonGradIn
  for (int i = 0; i < op_.numConvs(); i++) {
    // input at index getGradConvolvedIn(idx) : gradient of output of conv
    // input at index getWeightsIn(idx)       : weights input to conv
    inInfo.push_back({getGradConvolvedInIndex(i),
                      MultiConvBaseOp::getOutIndex(i),
                      GradOpInType::GradOut});
    inInfo.push_back({getWeightsInIndex(i),
                      MultiConvBaseOp::getWeightsInIndex(i),
                      GradOpInType::In});

    // the grad-op output at index i corresponds
    // to the conv ops input input index i
    gradOutInfo.emplace(getOutIndex(i), MultiConvBaseOp::getDataInIndex(i));

    // Set the output TensorInfo
    dataInfo.push_back(op_.inInfo(MultiConvBaseOp::getDataInIndex(i)));

    // Set the params for each convolution
    params.push_back(popx::getConvGradParameters(op_.getParameters(i)));
  }
}

void MultiConvDataGradBaseOp::setup() {
  for (int i = 0; i < dataInfo.size(); i++) {
    outInfo(getOutIndex(i)) = dataInfo.at(i);
  }
}

void MultiConvDataGradBaseOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);

  for (int64_t i = 0; i < numConvs(); i++) {
    std::string suffix = "[" + std::to_string(i) + "]";
    MultiConvBaseOp::appendConvParameterAttributes(params[i], suffix, os);

    // Append per-conv options
    for (auto key_val : getConvOptions().getConvOptions(i)) {
      os.appendAttribute(key_val.first + suffix, key_val.second);
    }
  }
}

} // namespace popart
