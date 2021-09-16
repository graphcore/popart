// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef NEURALNET_GUARD_POPX_OP_SCATTERUTILX_HPP
#define NEURALNET_GUARD_POPX_OP_SCATTERUTILX_HPP

#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>

namespace popart {
namespace popx {
namespace scatterutilx {

// poplin::linspace only supports float or half, this is for int
snap::Tensor linspace(snap::Graph &,
                      int left,
                      int right,
                      const poplar::DebugNameAndId &dnai,
                      int increment            = 1,
                      const poplar::Type &type = poplar::INT);

// Make b's rank match a.
//
// Assumes b.rank() <= a.rank() - dim
snap::Tensor matchRank(snap::Tensor, snap::Tensor, unsigned dim);

// Make b's shape match a.
//
// Assumes b is broadcastable into a
snap::Tensor broadcastShape(snap::Tensor, snap::Tensor);

void growScatter(poplar::program::Sequence &prog,
                 snap::Graph &,
                 const snap::Tensor &indices,
                 const snap::Tensor &replacementValues,
                 const snap::Tensor &dataToUpdateInPlace,
                 int64_t axis,
                 const poplar::DebugNameAndId &dnai);

snap::Tensor growScatterUpdateGrad(poplar::program::Sequence &prog,
                                   snap::Graph &graph,
                                   const snap::Tensor &gradIn,
                                   const snap::Tensor &indices,
                                   int64_t axis,
                                   const poplar::DebugNameAndId &dnai);
} // namespace scatterutilx
} // namespace popx
} // namespace popart

#endif
