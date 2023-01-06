// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SORTUTILX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SORTUTILX_HPP_

#include "popart/popx/debugcontextx.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
namespace popx {
namespace sortutilx {

poplar::Tensor getIotaTensor(poplar::Graph &graph,
                             const poplar::Tensor &input,
                             unsigned axis,
                             poplar::program::Sequence &prog,
                             const poplar::DebugNameAndId &dnai);

} // namespace sortutilx
} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SORTUTILX_HPP_
