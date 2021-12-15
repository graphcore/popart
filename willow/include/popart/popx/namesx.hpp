// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NAMESX_HPP
#define GUARD_NEURALNET_NAMESX_HPP

#include <poplar/Graph.hpp>
#include <popart/names.hpp>

namespace popart {
namespace popx {
// Pair of copy {source, target} tensor
using PreparedCopyTensor = std::pair<snap::Tensor, snap::Tensor>;
// IpuCopy input index to source and target tensor pair
using PreparedCopyTensors = std::map<InIndex, PreparedCopyTensor>;

} // namespace popx
} // namespace popart

#endif
