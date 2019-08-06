#ifndef GUARD_NEURALNET_TENSORNAMES_HPP
#define GUARD_NEURALNET_TENSORNAMES_HPP

#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {

// where tensor tenId is consumed by Op with OpId opId at
// index "index", what should the name of the edge-gradient
// along this edge be? This is pure string manipulation.
TensorId getEdgeGradId(TensorId tenId, OpId opId, int index);

// the name of the tensor which is the
// total gradient of a forward tensor
TensorId getGradId(const TensorId &tenId);

// inverse of previous function (non-grad name of grad tensor)
TensorId getNonGradId(const TensorId &tenId);

// get a recomputed tensor's name, based on original tensor
TensorId getRecompId(const TensorId &tenId);

// get an variable tensor's post-update name, based on original name
TensorId getUpdatedVarId(const TensorId &id);

constexpr const char *reservedGradientPrefix() { return "Gradient___"; }
constexpr const char *reservedUpdatedVarPrefix() { return "UpdatedVar___"; }
constexpr const char *reservedAccumulationPrefix() { return "Accumulation___"; }
constexpr const char *reservedAccumulationOutPrefix() {
  return "AccumulationOut___";
}
constexpr const char *reservedAccumulationResetPrefix() {
  return "AccumulationReset___";
}
constexpr const char *reservedStashedPrefix() { return "Stashed___"; }
constexpr const char *reservedRestoredPrefix() { return "Restored___"; }

std::vector<std::string> reservedPrefixes();

} // namespace popart

#endif
