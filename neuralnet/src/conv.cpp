#include <neuralnet/conv.hpp>
#include <neuralnet/error.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

ConvOp::ConvOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
    : Op(opId, node, pgraph), atts(node.attribute()){

  int nInputs = 0;
  for (auto &id : node.input()) {
    if (id != "") {
      ++nInputs;
    }
  }

  // handling this CONV as a special case, as it
  // needs splitting into 2 (CONV and add bias)
  // will still add, changed in optimise step.
  // this first step builds exactly 1-1 with onnx graph
  // and then later do the split
  if (nInputs == 3) {
    throw error("Conv with bias case not handled");
  }


}


void ConvOp::inferInfo() {

  nSpatialDims = input.tensor(0)->info.rank() - 2;
  batchSize = input.tensor(0)->info.dim(0);
  nInChans = input.tensor(0)->info.dim(1);
  nOutChans = input.tensor(1)->info.dim(0);

  // default values:
  dilations.resize(nSpatialDims, 1);
  group = 1;
  pads.resize(nSpatialDims, 1);
  strides.resize(nSpatialDims * 2, 1);

  // override defaults if onnx node stipulates:
  atts.setIfPresent(dilations, "dilations");
  atts.setIfPresent(group, "group");
  atts.setIfPresent(pads, "pads");
  atts.setIfPresent(strides, "strides");

  std::string auto_pad = "NOTSET";
  atts.setIfPresent(auto_pad, "auto_pad");
  if (auto_pad != "NOTSET"){
    throw error("auto_pad not NOTSET, deprecated and not supported");
  }

  std::vector<int64_t> outShape(2 + nSpatialDims, 0);
  outShape[0] = batchSize;
  outShape[1] = nOutChans;
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    // see https://pytorch.org/docs/stable/nn.html
    outShape[spDim + 2] =
        (input.tensor(0)->info.dim(spDim + 2) + pads[2 * spDim] +
         pads[2 * spDim + 1] -
         dilations[spDim] * (input.tensor(1)->info.dim(spDim + 2) - 1) - 1) /
            strides[spDim] +
        1;
  }

  output.tensor(0)->info.set(input.tensor(0)->info.dataType(), outShape);
}


} // namespace neuralnet
