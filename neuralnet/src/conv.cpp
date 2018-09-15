#include <neuralnet/conv.hpp>
#include <neuralnet/error.hpp>
#include <neuralnet/tensor.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <cblas.h>
#pragma clang diagnostic pop // stop ignoring warnings

namespace neuralnet {

ConvOp::ConvOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
    : HasReceptiveFieldOp(opId, node, pgraph) {
  if (input.n()) {
    throw error("Conv with bias case not handled");
  }
}

void ConvOp::setup0() {
  nOutChans = input.tensor(1)->info.dim(0);
  // setting groups from the input tensor,
  // we could also use the value in nAtts, as
  // "group" is required property of the ONNX conv op
  group = nInChans / input.tensor(1)->info.dim(1);
}

// ConvOp attributes only MIGHT contain the kernel shape,
// but we can ALWAYS get it directly from the kernel tensor
// at input index 1 so this is the preferred way to do it
void ConvOp::setSpatial() {
  spatial.reserve(nSpatialDims);
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    spatial.push_back(input.tensor(1)->info.dim(spDim + 2));
  }
}

int64_t ConvOp::getNOutChans() const { return nOutChans; }

} // namespace neuralnet
