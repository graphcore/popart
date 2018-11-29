#include <poponnx/error.hpp>
#include <poponnx/op/receptive.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

HasReceptiveFieldOp::HasReceptiveFieldOp(const onnx::NodeProto &node, Ir *_pir)
    : Op(node, _pir) {}

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

  // override defaults if onnx node stipulates:
  nAtts.setIfPresent(pads, "pads");
  nAtts.setIfPresent(strides, "strides");
  nAtts.setIfPresent(dilations, "dilations");

  std::string auto_pad = "NOTSET";
  nAtts.setIfPresent(auto_pad, "auto_pad");
  if (auto_pad != "NOTSET") {
    throw error("auto_pad not NOTSET, deprecated and not supported");
  }

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
  // see https://pytorch.org/docs/stable/nn.html for
  // determining spatial output size from conv parameters
  // it is essentially the same as for pooling
  // (same link, maxpool2d)
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    outShape[spDim + 2] =
        //(input.tensor(0)->info.dim(spDim + 2)
        (spatialD[spDim] + pads[spDim] + pads[nSpatialDims + spDim] -
         dilations[spDim] * (spatialK[spDim] - 1) - 1) /
            strides[spDim] +
        1;
  }

  return outShape;
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

} // namespace poponnx
