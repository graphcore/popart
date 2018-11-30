#ifndef GUARD_NEURALNET_TENSORNAMES_HPP
#define GUARD_NEURALNET_TENSORNAMES_HPP

#include <string>
#include <vector>
#include <poponnx/names.hpp>

namespace poponnx {

// where tensor tenId is consumed by Op with OpId opId at
// index "index", what should the name of the edge-gradient
// along this edge be? This is pure string manipulation.
TensorId getEdgeGradId(TensorId tenId, OpId opId, int index);

// the name of the tensor which is the
// total gradient of a forward tensor
TensorId getGradId(TensorId tenId);

// inverse of previous function (non-grad name of grad tensor)
TensorId getNonGradId(TensorId tenId);

// get a recomputed tensor's name, based on original tensor
TensorId getRecompId(TensorId tenId);

std::string reservedGradientPrefix();
std::string reservedRecomputePrefix();
std::vector<std::string> reservedPrefixes();

} // namespace poponnx

#endif
