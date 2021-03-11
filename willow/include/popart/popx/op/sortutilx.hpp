// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SORTUTILS_HPP
#define GUARD_NEURALNET_SORTUTILS_HPP

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

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

#endif
