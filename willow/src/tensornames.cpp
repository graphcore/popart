#include <sstream>
#include <poponnx/tensornames.hpp>

namespace poponnx {

TensorId getGradId(const TensorId &id) { return reservedGradientPrefix() + id; }

TensorId getRecompId(const TensorId &id) {
  return reservedRecomputePrefix() + id;
}

TensorId getUpdatedVarId(const TensorId &id) {
  return reservedUpdatedVarPrefix() + id;
}

TensorId getNonGradId(const TensorId &id) {
  // TODO : constexpr the size of this string T8265
  return id.substr(std::string(reservedGradientPrefix()).size());
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

std::vector<std::string> reservedPrefixes() {
  return {reservedGradientPrefix(),
          reservedRecomputePrefix(),
          reservedUpdatedVarPrefix()};
}

} // namespace poponnx
