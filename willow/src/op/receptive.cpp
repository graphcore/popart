// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <popart/error.hpp>
#include <popart/op/receptive.hpp>
#include <popart/opserialiser.hpp>
#include <popart/util.hpp>

#include "popart/attributes.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

HasReceptiveFieldOp::HasReceptiveFieldOp(
    const OperatorIdentifier &_opid,
    const HasReceptiveFieldOp::ReceptiveOpAttributes &attributes,
    const Op::Settings &settings_)
    : Op(_opid, settings_), basePads(attributes.pads),
      baseOutPads(attributes.outPads), baseStrides(attributes.strides),
      baseDilations(attributes.dilations),
      baseInDilations(attributes.inDilations),
      padType(getAutoPad(attributes.auto_pad)), ceilMode(attributes.ceil_mode) {
}

int HasReceptiveFieldOp::getNSpatialDims() const {
  auto &inputInfo = inInfo(getInIndex());
  return inputInfo.rank() - 2;
}

int64_t HasReceptiveFieldOp::getBatchSize() const {
  return inInfo(getInIndex()).dim(0);
}

int64_t HasReceptiveFieldOp::getNInChans() const {
  return inInfo(getInIndex()).dim(1);
}

std::vector<int64_t> HasReceptiveFieldOp::getSpatialD() const {
  auto nSpatialDims = getNSpatialDims();
  auto &inputInfo   = inInfo(getInIndex());

  std::vector<int64_t> spatialD(nSpatialDims);
  for (int i = 0; i < nSpatialDims; ++i) {
    spatialD[i] = inputInfo.dim(i + 2);
  }

  return spatialD;
}

std::vector<int64_t> HasReceptiveFieldOp::getSpatialO() const {
  auto nSpatialDims = getNSpatialDims();

  auto pads = basePads;
  pads.resize(nSpatialDims * 2, 0);
  Shape outShape = getOutShape(pads);

  std::vector<int64_t> spatialO(nSpatialDims);
  for (int i = 0; i < nSpatialDims; ++i) {
    spatialO[i] = outShape[i + 2];
  }

  return spatialO;
}

Shape HasReceptiveFieldOp::getOutPads() const {
  auto outPads = baseOutPads;
  outPads.resize(getNSpatialDims() * 2, 0);
  return outPads;
}

Shape HasReceptiveFieldOp::getPads() const {
  auto nSpatialDims = getNSpatialDims();

  auto pads = basePads;
  pads.resize(nSpatialDims * 2, 0);

  if (padType == AutoPad::SAME_LOWER || padType == AutoPad::SAME_UPPER) {
    alterPads(pads, getSpatialO(), getSpatialD(), getSpatialK(), getStrides());
  }

  if (ceilMode) {
    // Given the input parameters (pads, strides, dilations), if the calculated
    // output shape is non-integer, add upper-padding to each spatial dimension
    // such that the newly calculated output shape is the non-integer value
    // rounded up to the nearest integer.
    for (int i = 0; i < nSpatialDims; i++) {
      auto inferredUpperPad = ((getSpatialO()[i] - 1) * getStrides()[i]) +
                              getSpatialK()[i] - getSpatialD()[i] - pads[i];
      pads[i + nSpatialDims] = inferredUpperPad;
    }
  }

  return pads;
}

Shape HasReceptiveFieldOp::getStrides() const {
  auto nSpatialDims = getNSpatialDims();

  auto strides = baseStrides;
  strides.resize(nSpatialDims, 1);

  return strides;
}

Shape HasReceptiveFieldOp::getDilations() const {
  auto nSpatialDims = getNSpatialDims();

  auto dilations = baseDilations;
  dilations.resize(nSpatialDims, 1);

  return dilations;
}

Shape HasReceptiveFieldOp::getInDilations() const {
  int nSpatialDims = getNSpatialDims();

  auto inDilations = baseInDilations;
  inDilations.resize(nSpatialDims, 1);

  return inDilations;
}

void HasReceptiveFieldOp::setup() {
  // set-up whatever else the specific HasReceptiveFieldOp requires
  // need to be careful with the ordering here.
  setup0();

  // we assume that the output type is the same as the input type
  auto &inputInfo = inInfo(getInIndex());
  auto outType    = inputInfo.dataType();

  outInfo(getOutIndex()).set(outType, getOutShape(getPads()));
}

std::vector<int64_t> HasReceptiveFieldOp::lowerPads() const {
  return lowerPads(getPads(), getNSpatialDims(), padType);
}

std::vector<int64_t> HasReceptiveFieldOp::lowerPads(Shape pads_,
                                                    int nSpatialDims_,
                                                    AutoPad padType_) {
  if (padType_ == AutoPad::SAME_UPPER) {
    auto v = std::vector<int64_t>(pads_.begin(), pads_.begin() + nSpatialDims_);
    for (auto i = 0; i < v.size(); i++) {
      v[i] = int64_t(pads_[i] / 2);
    }
    return v;
  } else if (padType_ == AutoPad::SAME_LOWER) {
    auto v = std::vector<int64_t>(pads_.begin(), pads_.begin() + nSpatialDims_);
    for (auto i = 0; i < v.size(); i++) {
      v[i] = int64_t(pads_[i] - (pads_[i] / 2));
    }
    return v;
  } else {
    return std::vector<int64_t>(pads_.begin(), pads_.begin() + nSpatialDims_);
  }
}

std::vector<int64_t> HasReceptiveFieldOp::upperPads() const {
  return upperPads(getPads(), getNSpatialDims(), padType);
}

std::vector<int64_t> HasReceptiveFieldOp::upperPads(Shape pads_,
                                                    int nSpatialDims_,
                                                    AutoPad padType_) {

  // For odd values of total padding, add more padding at the 'right' side
  // of the given dimension.
  if (padType_ == AutoPad::SAME_UPPER) {
    auto v = std::vector<int64_t>(pads_.begin() + nSpatialDims_, pads_.end());
    for (auto i = 0; i < v.size(); i++) {
      v[i] = int64_t(pads_[i] - (pads_[i] / 2));
    }
    return v;
  } else if (padType_ == AutoPad::SAME_LOWER) {
    auto v = std::vector<int64_t>(pads_.begin() + nSpatialDims_, pads_.end());
    for (auto i = 0; i < v.size(); i++) {
      v[i] = int64_t(pads_[i] / 2);
    }
    return v;
  } else {
    return std::vector<int64_t>(pads_.begin() + nSpatialDims_, pads_.end());
  }
}

std::vector<int64_t> HasReceptiveFieldOp::lowerOutPads() const {
  auto outPads = getOutPads();
  return std::vector<int64_t>(outPads.begin(),
                              outPads.begin() + getNSpatialDims());
}

std::vector<int64_t> HasReceptiveFieldOp::upperOutPads() const {
  auto outPads = getOutPads();
  return std::vector<int64_t>(outPads.begin() + getNSpatialDims(),
                              outPads.end());
}

void HasReceptiveFieldOp::alterPads(Shape &pads_,
                                    Shape spatialO_,
                                    Shape spatialD_,
                                    Shape spatialK_,
                                    std::vector<int64_t> strides_) {
  for (auto i = 0; i < spatialO_.size(); i++) {
    pads_[i] = (spatialO_[i] - 1) * strides_[i] + spatialK_[i] - spatialD_[i];
  }
}

Shape HasReceptiveFieldOp::getOutShape(const Shape &pads) const {
  Shape outShape(2 + getNSpatialDims(), 0);
  outShape[0] = getBatchSize();
  outShape[1] = getNOutChans();

  Shape spatialOutShape = getSpatialOutShape(getSpatialD(),
                                             getSpatialK(),
                                             pads,
                                             getOutPads(),
                                             getStrides(),
                                             getDilations(),
                                             getInDilations(),
                                             padType,
                                             ceilMode);

  for (int spDim = 0; spDim < getNSpatialDims(); ++spDim) {
    outShape[spDim + 2] = spatialOutShape[spDim];
  }

  return outShape;
}

// see https://pytorch.org/docs/stable/nn.html for
// determining spatial output size from conv parameters
// it is essentially the same as for pooling
// (same link, maxpool2d)
Shape HasReceptiveFieldOp::getSpatialOutShape(Shape spatialD_,
                                              Shape spatialK_,
                                              std::vector<int64_t> pads_,
                                              std::vector<int64_t> outPads_,
                                              std::vector<int64_t> strides_,
                                              std::vector<int64_t> dilations_,
                                              std::vector<int64_t> inDilations_,
                                              AutoPad auto_pad_,
                                              bool ceil_mode_) {
  Shape spatialOutShape;
  int64_t numSpatialDims = spatialD_.size();

  auto round = [ceil_mode_](float dim) -> int64_t {
    if (ceil_mode_) {
      return std::ceil(dim);
    } else {
      return std::floor(dim);
    }
  };

  for (int spDim = 0; spDim < numSpatialDims; ++spDim) {
    int64_t dimSize = inDilations_[spDim] * (spatialD_[spDim] - 1) + 1;
    switch (auto_pad_) {
    case AutoPad::VALID: {
      dimSize = int(std::ceil(float(dimSize - (spatialK_[spDim] - 1)) /
                              float(strides_[spDim])));
      break;
    }
    case AutoPad::SAME_LOWER:
    case AutoPad::SAME_UPPER: {
      dimSize = int(std::ceil(float(dimSize) / float(strides_[spDim])));
      break;
    }
    case AutoPad::NOTSET: // default
    default: {
      dimSize += pads_[spDim] + pads_[numSpatialDims + spDim] - 1;
      dimSize -= dilations_[spDim] * (spatialK_[spDim] - 1);
      dimSize = int(round(float(dimSize) / float(strides_[spDim])));
      dimSize += 1;
    }
    }
    dimSize += outPads_[spDim] + outPads_[numSpatialDims + spDim];
    spatialOutShape.push_back(dimSize);
  }
  return spatialOutShape;
}

std::vector<size_t> HasReceptiveFieldOp::spatialD_szt() const {
  return vXtoY<int64_t, size_t>(getSpatialD());
}

std::vector<size_t> HasReceptiveFieldOp::spatialK_szt() const {
  return vXtoY<int64_t, size_t>(getSpatialK());
}

std::vector<uint32_t> HasReceptiveFieldOp::lowerPads_u32() const {
  return vXtoY<int64_t, uint32_t>(lowerPads());
}

std::vector<uint32_t> HasReceptiveFieldOp::upperPads_u32() const {
  return vXtoY<int64_t, uint32_t>(upperPads());
}

std::vector<int> HasReceptiveFieldOp::lowerPads_i32() const {
  return vXtoY<int64_t, int>(lowerPads());
}

std::vector<int> HasReceptiveFieldOp::upperPads_i32() const {
  return vXtoY<int64_t, int>(upperPads());
}

std::vector<uint32_t> HasReceptiveFieldOp::dilations_u32() const {
  return vXtoY<int64_t, uint32_t>(getDilations());
}

std::vector<uint32_t> HasReceptiveFieldOp::strides_u32() const {
  return vXtoY<int64_t, uint32_t>(getStrides());
}

void HasReceptiveFieldOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("basePads", basePads);
  os.appendAttribute("baseStrides", baseStrides);
  os.appendAttribute("baseDilations", baseDilations);
  os.appendAttribute("auto_pad", getAutoPadStr(padType));
}

AutoPad HasReceptiveFieldOp::getAutoPad(const std::string &autoPadStr) {

  if (autoPadStr == "SAME_UPPER")
    return AutoPad::SAME_UPPER;
  if (autoPadStr == "SAME_LOWER")
    return AutoPad::SAME_LOWER;
  if (autoPadStr == "VALID")
    return AutoPad::VALID;
  if (autoPadStr == "NOTSET")
    return AutoPad::NOTSET;
  if (autoPadStr == "")
    return AutoPad::NOTSET;
  throw error("Invalid auto_pad provied: {}", autoPadStr);
}

std::string HasReceptiveFieldOp::getAutoPadStr(const AutoPad &x) const {
  switch (x) {
  case AutoPad::VALID:
    return "VALID";
  case AutoPad::SAME_UPPER:
    return "SAME_UPPER";
  case AutoPad::SAME_LOWER:
    return "SAME_LOWER";
  case AutoPad::NOTSET:
    return "NOTSET";
  default:
    throw error("Bad AutoPad '{}'", static_cast<int>(x));
  }
}

void HasReceptiveFieldOp::ReceptiveOpAttributes::setFromAttributes(
    const Attributes &attributes) {
  attributes.setIfPresent(pads, "pads");
  attributes.setIfPresent(outPads, "outPads");
  attributes.setIfPresent(strides, "strides");
  attributes.setIfPresent(dilations, "dilations");
  attributes.setIfPresent(inDilations, "inDilations");
  attributes.setIfPresent(auto_pad, "auto_pad");
  attributes.setIfPresent(ceil_mode, "ceil_mode");
}

} // namespace popart
