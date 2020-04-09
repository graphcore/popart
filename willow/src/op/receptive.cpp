// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
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
      dilations(settings_.dilations) {}

void HasReceptiveFieldOp::setup() {

  auto &inputInfo = inInfo(getInIndex());

  batchSize    = inputInfo.dim(0);
  nInChans     = inputInfo.dim(1);
  nSpatialDims = inputInfo.rank() - 2;
  spatialD.resize(nSpatialDims);
  for (int i = 0; i < nSpatialDims; ++i) {
    spatialD[i] = inputInfo.dim(i + 2);
  }

  // default values:
  pads.resize(nSpatialDims * 2, 0);
  strides.resize(nSpatialDims, 1);
  dilations.resize(nSpatialDims, 1);

  setSpatialK();

  // set-up whatever else the specific HasReceptiveFieldOp requires
  // need to be careful with the ordering here.
  setup0();

  // we assume that the output type is the same as the input type
  outType = inputInfo.dataType();

  outInfo(getOutIndex()).set(outType, getOutShape());
}

std::vector<int64_t> HasReceptiveFieldOp::lowerPads() const {
  return std::vector<int64_t>(pads.begin(), pads.begin() + nSpatialDims);
}

std::vector<int64_t> HasReceptiveFieldOp::upperPads() const {
  return std::vector<int64_t>(pads.begin() + nSpatialDims, pads.end());
}

Shape HasReceptiveFieldOp::getOutShape() const {
  Shape outShape(2 + nSpatialDims, 0);
  outShape[0] = batchSize;
  outShape[1] = getNOutChans();

  Shape spatialOutShape =
      getSpatialOutShape(spatialD, spatialK, pads, strides, dilations);

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
                                              std::vector<int64_t> dilations_) {

  Shape spatialOutShape;
  int64_t numSpatialDims = spatialD_.size();

  for (int spDim = 0; spDim < numSpatialDims; ++spDim) {
    int64_t dimSize = spatialD_[spDim];
    dimSize += pads_[spDim] + pads_[numSpatialDims + spDim] - 1;
    dimSize -= dilations_[spDim] * (spatialK_[spDim] - 1);
    dimSize /= strides_[spDim];
    dimSize += 1;
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
}

void HasReceptiveFieldOp::Settings::setFromAttributes(
    const Attributes &attributes) {
  Op::Settings::setFromAttributes(attributes);

  attributes.setIfPresent(pads, "pads");
  attributes.setIfPresent(strides, "strides");
  attributes.setIfPresent(dilations, "dilations");

  if (attributes.hasAttribute("auto_pad")) {
    std::string autoPad =
        attributes.getAttribute<Attributes::String>("auto_pad", "NOTSET");
    if (autoPad != "NOTSET") {
      throw error(
          "auto_pad is set, but is deprecated and unsupported by popart");
    }
  }
}

} // namespace popart
