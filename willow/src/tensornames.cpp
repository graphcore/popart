#include <sstream>
#include <poponnx/tensornames.hpp>

namespace poponnx {

TensorId getGradId(TensorId id) { return reservedGradientPrefix() + id; }

TensorId getRecompId(TensorId id) { return reservedRecomputePrefix() + id; }

TensorId getNonGradId(TensorId id) {
  return id.substr(reservedGradientPrefix().size());
}

TensorId getEdgeGradId(TensorId tenId, OpId opId, int index) {
  // we don't need the name of the tensor which this is an edge-grad of,
  // the edge-gradient is uniquely defined by the the edge it flows on
  // in the forward pass (input at 'index' to 'opId')
  (void)tenId;
  std::stringstream ss;
  ss << reservedGradientPrefix() << opId << '_' << index;
  TensorId edgeGradId = ss.str();
  return edgeGradId;
}

} // namespace poponnx
