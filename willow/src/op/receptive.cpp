// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <math.h>
#include <memory>
#include <unordered_map>
#include <popart/error.hpp>
#include <popart/op/receptive.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

HasReceptiveFieldOp::HasReceptiveFieldOp(
    const OperatorIdentifier &_opid,
    const HasReceptiveFieldOp::Settings &settings_)
    : Op(_opid, settings_), pads(settings_.pads), strides(settings_.strides),
      dilations(settings_.dilations), padType(getAutoPad(settings_.auto_pad)),
      ceilMode(settings_.ceil_mode) {}

void HasReceptiveFieldOp::setup() {

  auto &inputInfo = inInfo(getInIndex());

  batchSize    = inputInfo.dim(0);
  nInChans     = inputInfo.dim(1);
  nSpatialDims = inputInfo.rank() - 2;
  spatialD.resize(nSpatialDims);
  for (int i = 0; i < nSpatialDims; ++i) {
    spatialD[i] = inputInfo.dim(i + 2);
  }
  spatialO.resize(nSpatialDims);

  // default values:
  pads.resize(nSpatialDims * 2, 0);
  strides.resize(nSpatialDims, 1);
  dilations.resize(nSpatialDims, 1);

  setSpatialK();

  // set-up whatever else the specific HasReceptiveFieldOp requires
  // need to be careful with the ordering here.
  setup0();

  Shape outShape = getOutShape();
  for (int i = 0; i < nSpatialDims; ++i) {
    spatialO[i] = outShape[i + 2];
  }

  if (padType == AutoPad::SAME_LOWER || padType == AutoPad::SAME_UPPER) {
    alterPads(pads, spatialO, spatialD, spatialK, strides);
  }

  if (ceilMode) {
    // Given the input parameters (pads, strides, dilations), if the calculated
    // output shape is non-integer, add upper-padding to each spatial dimension
    // such that the newly calculated output shape is the non-integer value
    // rounded up to the nearest integer.
    for (int i = 0; i < nSpatialDims; i++) {
      auto inferredUpperPad = ((spatialO[i] - 1) * strides[i]) + spatialK[i] -
                              spatialD[i] - pads[i];
      pads[i + nSpatialDims] = inferredUpperPad;
    }
  }

  // we assume that the output type is the same as the input type
  outType = inputInfo.dataType();

  outInfo(getOutIndex()).set(outType, getOutShape());
}

std::vector<int64_t> HasReceptiveFieldOp::lowerPads() const {
  return lowerPads(pads, nSpatialDims, padType);
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
  return upperPads(pads, nSpatialDims, padType);
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

void HasReceptiveFieldOp::alterPads(Shape &pads_,
                                    Shape spatialO_,
                                    Shape spatialD_,
                                    Shape spatialK_,
                                    std::vector<int64_t> strides_) {
  for (auto i = 0; i < spatialO_.size(); i++) {
    pads_[i] = (spatialO_[i] - 1) * strides_[i] + spatialK_[i] - spatialD_[i];
  }
}

Shape HasReceptiveFieldOp::getOutShape() const {
  Shape outShape(2 + nSpatialDims, 0);
  outShape[0] = batchSize;
  outShape[1] = getNOutChans();

  Shape spatialOutShape = getSpatialOutShape(
      spatialD, spatialK, pads, strides, dilations, padType, ceilMode);

  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
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
                                              std::vector<int64_t> strides_,
                                              std::vector<int64_t> dilations_,
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
    int64_t dimSize;
    switch (auto_pad_) {
    case AutoPad::VALID: {
      dimSize = int(std::ceil(float(spatialD_[spDim] - (spatialK_[spDim] - 1)) /
                              float(strides_[spDim])));
      break;
    }
    case AutoPad::SAME_LOWER:
    case AutoPad::SAME_UPPER: {
      dimSize =
          int(std::ceil(float(spatialD_[spDim]) / float(strides_[spDim])));
      break;
    }
    case AutoPad::NOTSET: // default
    default: {
      dimSize = spatialD_[spDim];
      dimSize += pads_[spDim] + pads_[numSpatialDims + spDim] - 1;
      dimSize -= dilations_[spDim] * (spatialK_[spDim] - 1);
      dimSize = int(round(float(dimSize) / float(strides_[spDim])));
      dimSize += 1;
    }
    }
    spatialOutShape.push_back(dimSize);
  }
  return spatialOutShape;
}

std::vector<size_t> HasReceptiveFieldOp::spatialD_szt() const {
  return vXtoY<int64_t, size_t>(spatialD);
}

std::vector<size_t> HasReceptiveFieldOp::spatialK_szt() const {
  return vXtoY<int64_t, size_t>(spatialK);
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
  return vXtoY<int64_t, uint32_t>(dilations);
}

std::vector<uint32_t> HasReceptiveFieldOp::strides_u32() const {
  return vXtoY<int64_t, uint32_t>(strides);
}

void HasReceptiveFieldOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("pads", pads);
  os.appendAttribute("strides", strides);
  os.appendAttribute("dilations", dilations);
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

void HasReceptiveFieldOp::Settings::setFromAttributes(
    const Attributes &attributes) {
  Op::Settings::setFromAttributes(attributes);

  attributes.setIfPresent(pads, "pads");
  attributes.setIfPresent(strides, "strides");
  attributes.setIfPresent(dilations, "dilations");
  attributes.setIfPresent(auto_pad, "auto_pad");
  attributes.setIfPresent(ceil_mode, "ceil_mode");
}

} // namespace popart
