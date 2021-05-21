// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef NEURALNET_GUARD_POPX_OP_SCATTERUTILX_HPP
#define NEURALNET_GUARD_POPX_OP_SCATTERUTILX_HPP

#include <poplar/Tensor.hpp>

#include <snap/Graph.hpp>

namespace popart {
namespace popx {
namespace scatterutilx {

// poplin::linspace only supports float or half, this is for int
poplar::Tensor linspace(snap::Graph &,
                        int left,
                        int right,
                        const poplar::DebugNameAndId &dnai,
                        int increment            = 1,
                        const poplar::Type &type = poplar::INT);

// Make b's rank match a.
//
// Assumes b.rank() <= a.rank() - dim
poplar::Tensor matchRank(poplar::Tensor, poplar::Tensor, unsigned dim);

// Make b's shape match a.
//
// Assumes b is broadcastable into a
poplar::Tensor broadcastShape(poplar::Tensor, poplar::Tensor);

void growScatter(poplar::program::Sequence &prog,
                 snap::Graph &,
                 const poplar::Tensor &indices,
                 const poplar::Tensor &replacementValues,
                 const poplar::Tensor &dataToUpdateInPlace,
                 int64_t axis,
                 const poplar::DebugNameAndId &dnai);

poplar::Tensor growScatterUpdateGrad(poplar::program::Sequence &prog,
                                     snap::Graph &graph,
                                     const poplar::Tensor &gradIn,
                                     const poplar::Tensor &indices,
                                     int64_t axis,
                                     const poplar::DebugNameAndId &dnai);
} // namespace scatterutilx
} // namespace popx
} // namespace popart

#endif
