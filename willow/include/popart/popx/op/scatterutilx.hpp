// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERUTILX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERUTILX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <cstdint>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popart/names.hpp>

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popops {
class SlicePlan;
}

namespace popart {
class TensorInfo;

namespace popx {
class Opx;

namespace scatterutilx {

// poplin::linspace only supports float or half, this is for int
poplar::Tensor linspace(poplar::Graph &,
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

// Linearize the indices: map from 2-d indices to 1-d
poplar::Tensor linearizeIndices(const Opx &opx,
                                poplar::program::Sequence &prog,
                                poplar::Tensor indices,
                                int numDataCols,
                                unsigned group_size);

void growScatter(poplar::program::Sequence &prog,
                 poplar::Graph &,
                 const poplar::Tensor &indices,
                 const poplar::Tensor &replacementValues,
                 const poplar::Tensor &dataToUpdateInPlace,
                 int64_t axis,
                 const poplar::DebugNameAndId &dnai);

poplar::Tensor growScatterUpdateGrad(const Opx &opx,
                                     poplar::program::Sequence &prog,
                                     poplar::Graph &graph,
                                     const poplar::Tensor &gradIn,
                                     const poplar::Tensor &indicesIn,
                                     const popart::Shape &gradOutShape,
                                     int64_t axis,
                                     const popops::SlicePlan &plan,
                                     const poplar::DebugNameAndId &dnai);

} // namespace scatterutilx
} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCATTERUTILX_HPP_
