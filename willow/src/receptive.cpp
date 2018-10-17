#include <willow/error.hpp>
#include <willow/receptive.hpp>
#include <willow/tensor.hpp>

namespace willow {

HasReceptiveFieldOp::HasReceptiveFieldOp(const onnx::NodeProto &node, Ir *pir)
    : Op(node, pir) {}

void HasReceptiveFieldOp::setup() {
  batchSize    = input.tensor(0)->info.dim(0);
  nInChans     = input.tensor(0)->info.dim(1);
  nSpatialDims = input.tensor(0)->info.rank() - 2;

  // default values:
  pads.resize(nSpatialDims, 0);
  strides.resize(nSpatialDims * 2, 1);
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

  setSpatial();

  // set-up whatever else the specific HasReceptiveFieldOp requires
  // need to be careful with the ordering here.
  setup0();

  output.tensor(0)->info.set(input.tensor(0)->info.dataType(), getOutShape());
}

std::vector<int64_t> HasReceptiveFieldOp::getOutShape() const {
  std::vector<int64_t> outShape(2 + nSpatialDims, 0);
  outShape[0] = batchSize;
  outShape[1] = getNOutChans();
  // see https://pytorch.org/docs/stable/nn.html for
  // determining spatial output size from conv parameters
  // it is essentially the same as for pooling
  // (same link, maxpool2d)
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    outShape[spDim + 2] =
        (input.tensor(0)->info.dim(spDim + 2) + pads[2 * spDim] +
         pads[2 * spDim + 1] - dilations[spDim] * (spatial[spDim] - 1) - 1) /
            strides[spDim] +
        1;
  }

  return outShape;
}
} // namespace willow
