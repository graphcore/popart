// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_NAMESX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_NAMESX_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_NAMESX_HPP_
