// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SORTUTILS_HPP
#define GUARD_NEURALNET_SORTUTILS_HPP

#include <poplar/DebugContext.hpp>
#include <poplar/Program.hpp>

#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>

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

#endif
