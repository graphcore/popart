#include <neuralnet/averagepool.hpp>
#include <neuralnet/error.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

AveragePoolOp::AveragePoolOp(OpId opId,
                             const onnx::NodeProto &node,
                             Graph *pgraph)
    : HasReceptiveFieldOp(opId, node, pgraph) {}

void AveragePoolOp::setup0() {}

void AveragePoolOp::setSpatial() {
  spatial.reserve(nSpatialDims);
  std::vector<int64_t> kernel_shape;
  nAtts.setIfPresent(kernel_shape, "kernel_shape");
  if (kernel_shape.size() != input.tensor(0)->info.rank() - 2) {
    throw error("invald kernel_shape, not same rank as tensor operate on");
  }
  for (int spDim = 0; spDim < nSpatialDims; ++spDim) {
    spatial.push_back(kernel_shape[spDim]);
  }
}

// Pooling does not change the number of channels,
// i.e it is the same as the number of input channels
int64_t AveragePoolOp::getNOutChans() const { return nInChans; }
} // namespace neuralnet
