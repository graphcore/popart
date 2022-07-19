// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SORTUTILX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SORTUTILX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <snap/Tensor.hpp>

namespace snap {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
namespace popx {
namespace sortutilx {

snap::Tensor getIotaTensor(snap::Graph &graph,
                           const snap::Tensor &input,
                           unsigned axis,
                           snap::program::Sequence &prog,
                           const poplar::DebugNameAndId &dnai);

} // namespace sortutilx
} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SORTUTILX_HPP_
