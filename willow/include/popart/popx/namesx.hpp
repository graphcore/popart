// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_NAMESX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_NAMESX_HPP_

#include <map>
#include <utility>
#include <popart/names.hpp>

namespace poplar {
class Tensor;
} // namespace poplar

namespace popart {
namespace popx {
// Pair of copy {source, target} tensor
using PreparedCopyTensor = std::pair<poplar::Tensor, poplar::Tensor>;
// IpuCopy input index to source and target tensor pair
using PreparedCopyTensors = std::map<InIndex, PreparedCopyTensor>;

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_NAMESX_HPP_
