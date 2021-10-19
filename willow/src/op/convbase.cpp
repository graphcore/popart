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

std::ostream &operator<<(std::ostream &os, const ConvParameters::Input &input) {
  os << "[lowerTruncation: " << input.lowerTruncation
     << " upperTruncation: " << input.upperTruncation
     << " dilation: " << input.dilation
     << " lowerPadding: " << input.lowerPadding
     << " upperPadding: " << input.upperPadding << " flip: [";

  for (auto it = input.flip.begin(); it != input.flip.end(); it++) {
    os << static_cast<bool>(*it);
    if (it + 1 != input.flip.end()) {
      os << " ";
    }
  }

  os << "]]";
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const ConvParameters::Output &input) {
  os << "[lowerTruncation: " << input.lowerTruncation
     << " upperTruncation: " << input.upperTruncation
     << " stride: " << input.stride << " lowerPadding: " << input.lowerPadding
     << " upperPadding: " << input.upperPadding << "]";

  return os;
}

std::ostream &operator<<(std::ostream &os, const ConvParameters &params) {
  os << "type: " << params.type << " batchSize: " << params.batchSize
     << " numInChannelsPerGroup: " << params.numInChannelsPerGroup
     << " numOutChannelsPerGroup: " << params.numOutChannelsPerGroup
     << " numGroups: " << params.numGroups
     << " inputShape: " << params.inputShape
     << " kernelShape :" << params.kernelShape
     << " inputTransformation: " << params.inputTransformation
     << " kernelTransformation: " << params.kernelTransformation
     << " outputTransformation: " << params.outputTransformation;

  return os;
}

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

  if (attr.hasAttribute(sEnableConvDitheringAttribute)) {
    // Set from locally set attribute
    if (attr.getAttribute<Attributes::Ints>(sEnableConvDitheringAttribute)
            .size()) {
      // > 1 convolution
      enableConvDithering =
          attr.getAttribute<Attributes::Ints>(sEnableConvDitheringAttribute);
    } else {
      // only one convolution
      enableConvDithering = {
          attr.getAttribute<Attributes::Int>(sEnableConvDitheringAttribute)};
    }
  } else {
    // Set from session-wide convolution settings
    auto convDitheringOpt = sessionConvOptions.find("enableConvDithering");
    if (convDitheringOpt != sessionConvOptions.end()) {
      enableConvDithering =
          std::vector<int64_t>(numConvs, std::stoll(convDitheringOpt->second));
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
  if (enableConvDithering.size()) {
    strings["enableConvDithering"] =
        enableConvDithering[convIndex] != 0 ? "true" : "false";
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

Shape MultiConvBaseOp::getOutShape(int convIndex, const ConvPads &pads) const {
  const auto nSpatialDims = getNSpatialDims(convIndex);
  Shape outShape(2 + nSpatialDims, 0);
  outShape[0] = inInfo(getDataInIndex(convIndex)).dim(0); // batch size
  outShape[1] = getNOutChans(convIndex);

  Shape kernSpatialD = getSpatialK(convIndex);

  // Take into account any kenel truncation. an example of this is calculating
  // the gradient of a convolution whose kernel covered only padding and not
  // acutual inputs.
  Shape lowKernTruncs = lowerKernTruncs(convIndex);
  Shape upKernTruncs  = upperKernTruncs(convIndex);
  for (size_t dim = 0; dim < nSpatialDims; dim++) {
    kernSpatialD[dim] -= lowKernTruncs[dim];
    kernSpatialD[dim] -= upKernTruncs[dim];
  }

  Shape inSpatialD = getSpatialD(convIndex);

  // Take into account any input trunction. An example of this is calculating
  // the data gradient of a convolution whose padding exceeds its kernel size.
  Shape lowInTruncs = lowerInTruncs(convIndex);
  Shape upInTruncs  = upperInTruncs(convIndex);
  for (size_t dim = 0; dim < nSpatialDims; dim++) {
    inSpatialD[dim] -= lowInTruncs[dim];
    inSpatialD[dim] -= upInTruncs[dim];
  }

  Shape spatialOutShape =
      HasReceptiveFieldOp::getSpatialOutShape(inSpatialD,
                                              kernSpatialD,
                                              pads,
                                              getOutPads(convIndex),
                                              getStrides(convIndex),
                                              getDilations(convIndex),
                                              getInDilations(convIndex),
                                              padType);

  // Take into account any output truncation.
  Shape lowOutTruncs = lowerOutTruncs(convIndex);
  Shape upOutTruncs  = upperOutTruncs(convIndex);

  for (size_t dim = 0; dim < nSpatialDims; dim++) {
    spatialOutShape[dim] -= lowOutTruncs[dim];
    spatialOutShape[dim] -= upOutTruncs[dim];
  }

  // Combine to make the full output shape
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    outShape[spDim + 2] = spatialOutShape[spDim];
  }
  return outShape;
}

void MultiConvBaseOp::setParamsFromDataGradOp(const Op *op) {
  // Set output shape and parameters
  auto dataGradOp = dynamic_cast<const MultiConvDataGradBaseOp *>(op);
  std::vector<ConvParameters> ps;
  for (int i = 0; i < numConvs(); i++) {
    ps.push_back(dataGradOp->getParameters(i));
  }
  restoreAttributesFromParams(ps);
}

namespace {
void ensureZeros(const std::vector<int64_t> &vals, const char *name) {
  for (auto val : vals) {
    if (val != 0) {
      std::stringstream ss;
      ss << "non-zero param." << name << " is not supported in "
         << "MultiConvBaseOp.";
      throw error(ss.str());
    }
  }
}

void ensureFalses(const std::vector<bool> &vals, const char *name) {
  for (auto val : vals) {
    if (val) {
      std::stringstream ss;
      ss << "true param." << name << " is not supported in "
         << "MultiConvBaseOp.";
      throw error(ss.str());
    }
  }
}
} // namespace

void MultiConvBaseOp::restoreAttributesFromParams(
    const std::vector<ConvParameters> &ps) {
  // Restore Op parameters from Conv parameters so that setup() works
  flatStrides.clear();
  flatPads.clear();
  flatOutPads.clear();
  flatDilations.clear();
  flatInDilations.clear();

  flatKernTruncs.clear();

  flatInTruncs.clear();
  flatOutTruncs.clear();

  for (auto param : ps) {
    // inputTransformation
    flatPads.insert(flatPads.end(),
                    param.inputTransformation.lowerPadding.begin(),
                    param.inputTransformation.lowerPadding.end());
    flatPads.insert(flatPads.end(),
                    param.inputTransformation.upperPadding.begin(),
                    param.inputTransformation.upperPadding.end());
    flatInDilations.insert(flatInDilations.end(),
                           param.inputTransformation.dilation.begin(),
                           param.inputTransformation.dilation.end());
    flatInTruncs.insert(flatInTruncs.end(),
                        param.inputTransformation.lowerTruncation.begin(),
                        param.inputTransformation.lowerTruncation.end());
    flatInTruncs.insert(flatInTruncs.end(),
                        param.inputTransformation.upperTruncation.begin(),
                        param.inputTransformation.upperTruncation.end());
    ensureFalses(param.inputTransformation.flip, "inputTransformation.flip");

    // kernelTransformation
    flatKernTruncs.insert(flatKernTruncs.end(),
                          param.kernelTransformation.lowerTruncation.begin(),
                          param.kernelTransformation.lowerTruncation.end());
    flatKernTruncs.insert(flatKernTruncs.end(),
                          param.kernelTransformation.upperTruncation.begin(),
                          param.kernelTransformation.upperTruncation.end());

    flatDilations.insert(flatDilations.end(),
                         param.kernelTransformation.dilation.begin(),
                         param.kernelTransformation.dilation.end());

    ensureZeros(param.kernelTransformation.lowerPadding,
                "kernelTransformation.lowerPadding");
    ensureZeros(param.kernelTransformation.upperPadding,
                "kernelTransformation.upperPadding");
    ensureFalses(param.kernelTransformation.flip, "kernelTransformation.flip");

    flatStrides.insert(flatStrides.end(),
                       param.outputTransformation.stride.begin(),
                       param.outputTransformation.stride.end());

    flatOutPads.insert(flatOutPads.end(),
                       param.outputTransformation.lowerPadding.begin(),
                       param.outputTransformation.lowerPadding.end());
    flatOutPads.insert(flatOutPads.end(),
                       param.outputTransformation.upperPadding.begin(),
                       param.outputTransformation.upperPadding.end());
    flatOutTruncs.insert(flatOutTruncs.end(),
                         param.outputTransformation.lowerTruncation.begin(),
                         param.outputTransformation.lowerTruncation.end());
    flatOutTruncs.insert(flatOutTruncs.end(),
                         param.outputTransformation.upperTruncation.begin(),
                         param.outputTransformation.upperTruncation.end());
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

    appendConvParameterAttributes(getParameters(i), suffix, os);

    // Append per-conv options
    for (auto key_val : getConvOptions().getConvOptions(i)) {
      os.appendAttribute(key_val.first + suffix, key_val.second);
    }
  }
}

int64_t MultiConvBaseOp::getCumulativeSpatialDims(int64_t convIndex) const {
  int64_t cumulativeSpatialDims = 0;

  for (int i = 0; i < convIndex; i++) {
    cumulativeSpatialDims += getNSpatialDims(i);
  }

  return cumulativeSpatialDims;
}

ConvStrides MultiConvBaseOp::getStrides(int64_t convIndex) const {
  const auto nSpatialDims          = getNSpatialDims(convIndex);
  const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);

  ConvStrides result;
  if (flatStrides.empty()) {
    result.resize(nSpatialDims, 1);
  } else {
    result = {flatStrides.begin() + cumulativeSpatialDims,
              flatStrides.begin() + cumulativeSpatialDims + nSpatialDims};
  }

  return result;
}

ConvPads MultiConvBaseOp::getPads(int64_t convIndex) const {
  const auto nSpatialDims          = getNSpatialDims(convIndex);
  const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);

  ConvPads pads;
  if (flatPads.empty()) {
    pads.resize(2 * nSpatialDims, 0);
  } else {
    const auto cumulativePads = cumulativeSpatialDims * 2;
    pads                      = {flatPads.begin() + cumulativePads,
            flatPads.begin() + cumulativePads + (nSpatialDims * 2)};
  }

  // Alter pads if necessary, based on AutoPad type
  if (padType == AutoPad::SAME_LOWER || padType == AutoPad::SAME_UPPER) {
    const auto outShape = getOutShape(convIndex, pads);
    Shape spatialO(nSpatialDims, 0);
    for (int j = 0; j < nSpatialDims; ++j) {
      spatialO[j] = outShape[j + 2];
    }

    HasReceptiveFieldOp::alterPads(pads,
                                   spatialO,
                                   getSpatialD(convIndex),
                                   getSpatialK(convIndex),
                                   getStrides(convIndex));
  }

  return pads;
}

ConvPads MultiConvBaseOp::getOutPads(int64_t convIndex) const {
  const auto nSpatialDims          = getNSpatialDims(convIndex);
  const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);

  ConvPads result;
  if (flatOutPads.empty()) {
    result.resize(2 * nSpatialDims, 0);
  } else {
    const auto cumulativePads = cumulativeSpatialDims * 2;
    result                    = {flatOutPads.begin() + cumulativePads,
              flatOutPads.begin() + cumulativePads + (nSpatialDims * 2)};
  }
  return result;
}

ConvDilations MultiConvBaseOp::getDilations(int64_t convIndex) const {
  const auto nSpatialDims          = getNSpatialDims(convIndex);
  const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);

  ConvDilations result;
  if (flatDilations.empty()) {
    result.resize(nSpatialDims, 1);
  } else {
    result = {flatDilations.begin() + cumulativeSpatialDims,
              flatDilations.begin() + cumulativeSpatialDims + nSpatialDims};
  }
  return result;
}

ConvDilations MultiConvBaseOp::getInDilations(int64_t convIndex) const {
  const auto nSpatialDims          = getNSpatialDims(convIndex);
  const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);

  ConvDilations result;
  if (flatInDilations.empty()) {
    result.resize(nSpatialDims, 1);
  } else {
    result = {flatInDilations.begin() + cumulativeSpatialDims,
              flatInDilations.begin() + cumulativeSpatialDims + nSpatialDims};
  }
  return result;
}

Shape MultiConvBaseOp::lowerKernTruncs(int64_t convIndex) const {
  const auto nSpatialDims = getNSpatialDims(convIndex);

  Shape result;
  if (flatKernTruncs.empty()) {
    result.resize(nSpatialDims, 0);
  } else {
    const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);
    const auto cumulativeTruncs      = cumulativeSpatialDims * 2;
    result = {flatKernTruncs.begin() + cumulativeTruncs,
              flatKernTruncs.begin() + cumulativeTruncs + nSpatialDims};
  }
  return result;
}

Shape MultiConvBaseOp::upperKernTruncs(int64_t convIndex) const {
  const auto nSpatialDims = getNSpatialDims(convIndex);

  Shape result;
  if (flatKernTruncs.empty()) {
    result.resize(nSpatialDims, 0);
  } else {
    const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);
    const auto cumulativeTruncs      = cumulativeSpatialDims * 2;
    result = {flatKernTruncs.begin() + cumulativeTruncs + nSpatialDims,
              flatKernTruncs.begin() + cumulativeTruncs + nSpatialDims * 2};
  }
  return result;
}

Shape MultiConvBaseOp::lowerInTruncs(int64_t convIndex) const {
  const auto nSpatialDims = getNSpatialDims(convIndex);

  Shape result;
  if (flatInTruncs.empty()) {
    result.resize(nSpatialDims, 0);
  } else {
    const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);
    const auto cumulativeTruncs      = cumulativeSpatialDims * 2;
    result                           = {flatInTruncs.begin() + cumulativeTruncs,
              flatInTruncs.begin() + cumulativeTruncs + nSpatialDims};
  }
  return result;
}

Shape MultiConvBaseOp::upperInTruncs(int64_t convIndex) const {
  const auto nSpatialDims = getNSpatialDims(convIndex);

  Shape result;
  if (flatInTruncs.empty()) {
    result.resize(nSpatialDims, 0);
  } else {
    const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);
    const auto cumulativeTruncs      = cumulativeSpatialDims * 2;
    result = {flatInTruncs.begin() + cumulativeTruncs + nSpatialDims,
              flatInTruncs.begin() + cumulativeTruncs + nSpatialDims * 2};
  }
  return result;
}

Shape MultiConvBaseOp::lowerOutTruncs(int64_t convIndex) const {
  const auto nSpatialDims = getNSpatialDims(convIndex);

  Shape result;
  if (flatOutTruncs.empty()) {
    result.resize(nSpatialDims, 0);
  } else {
    const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);
    const auto cumulativeTruncs      = cumulativeSpatialDims * 2;
    result = {flatOutTruncs.begin() + cumulativeTruncs,
              flatOutTruncs.begin() + cumulativeTruncs + nSpatialDims};
  }
  return result;
}

Shape MultiConvBaseOp::upperOutTruncs(int64_t convIndex) const {
  const auto nSpatialDims = getNSpatialDims(convIndex);

  Shape result;
  if (flatOutTruncs.empty()) {
    result.resize(nSpatialDims, 0);
  } else {
    const auto cumulativeSpatialDims = getCumulativeSpatialDims(convIndex);
    const auto cumulativeTruncs      = cumulativeSpatialDims * 2;
    result = {flatOutTruncs.begin() + cumulativeTruncs + nSpatialDims,
              flatOutTruncs.begin() + cumulativeTruncs + nSpatialDims * 2};
  }
  return result;
}

void MultiConvBaseOp::setup() {
  checkParameters();

  // Set output shapes
  for (int i = 0; i < numConvs(); i++) {
    outInfo(getOutIndex(i))
        .set(inInfo(getDataInIndex(i)).dataType(), getOutShape(i, getPads(i)));
  }
}

void MultiConvBaseOp::checkParameters() const {
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
}

ConvParameters MultiConvBaseOp::getParameters(int convIndex) const {
  const auto nSpatialDims = getNSpatialDims(convIndex);
  std::vector<int64_t> zeros(nSpatialDims, 0);
  std::vector<int64_t> ones(nSpatialDims, 1);
  std::vector<bool> falses(nSpatialDims, false);

  ConvParameters result;
  result.type        = inInfo(getDataInIndex(convIndex)).dataType();
  result.batchSize   = inInfo(getDataInIndex(convIndex)).dim(0);
  result.inputShape  = getSpatialD(convIndex);
  result.kernelShape = getSpatialK(convIndex);
  result.numInChannelsPerGroup =
      inInfo(getDataInIndex(convIndex)).dim(1) / getGroups(convIndex);
  result.numOutChannelsPerGroup =
      getNOutChans(convIndex) / getGroups(convIndex);
  result.numGroups = getGroups(convIndex);

  result.inputTransformation.lowerTruncation = lowerInTruncs(convIndex);
  result.inputTransformation.upperTruncation = upperInTruncs(convIndex);
  result.inputTransformation.dilation        = getInDilations(convIndex);
  result.inputTransformation.lowerPadding    = lowerPads(convIndex);
  result.inputTransformation.upperPadding    = upperPads(convIndex);
  result.inputTransformation.flip            = falses;

  result.kernelTransformation.lowerTruncation = lowerKernTruncs(convIndex);
  result.kernelTransformation.upperTruncation = upperKernTruncs(convIndex);
  result.kernelTransformation.dilation        = getDilations(convIndex);
  result.kernelTransformation.lowerPadding    = zeros;
  result.kernelTransformation.upperPadding    = zeros;
  result.kernelTransformation.flip            = falses;

  result.outputTransformation.lowerTruncation = lowerOutTruncs(convIndex);
  result.outputTransformation.upperTruncation = upperOutTruncs(convIndex);
  result.outputTransformation.stride          = getStrides(convIndex);
  result.outputTransformation.lowerPadding    = lowerOutPads(convIndex);
  result.outputTransformation.upperPadding    = upperOutPads(convIndex);

  return result;
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
      getIr().getSessionOptions().batchSerializationSettings.factor < 2 &&
      !getIr().getSessionOptions().explicitRecomputation) {
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
