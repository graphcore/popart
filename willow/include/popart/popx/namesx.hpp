// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NAMESX_HPP
#define GUARD_NEURALNET_NAMESX_HPP

#include <map>
#include <utility>
#include <popart/names.hpp>

namespace snap {
class Tensor;
} // namespace snap

namespace popart {
namespace popx {
// Pair of copy {source, target} tensor
using PreparedCopyTensor = std::pair<snap::Tensor, snap::Tensor>;
// IpuCopy input index to source and target tensor pair
using PreparedCopyTensors = std::map<InIndex, PreparedCopyTensor>;

} // namespace popx
} // namespace popart

#endif
