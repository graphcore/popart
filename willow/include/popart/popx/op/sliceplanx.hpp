// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SLICEPLANX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SLICEPLANX_HPP_

#include <cstddef>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <popops/DynamicSlice.hpp>
#include <popart/names.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace poplar {
class Graph;
}

namespace popart {
class TensorInfo;

namespace popx {

enum class SlicePlanUsedFor {
  Slice,
  Update,
  UpdateAdd,
  UpdateMax,
  CombinedSliceUpdate
};

poplar::OptionFlags
createSlicePlanOptions(SlicePlanUsedFor usedFor,
                       nonstd::optional<float> availableMemoryProportion = {});

popops::SlicePlan createSlicePlan(const poplar::Graph &graph,
                                  const popart::TensorInfo &sliceableInfo,
                                  const popart::TensorInfo &indicesInfo,
                                  const poplar::OptionFlags &options,
                                  nonstd::optional<size_t> axis = {},
                                  size_t group_size             = 1);

poplar::Tensor createDataTensor(poplar::Graph &graph,
                                const popart::TensorInfo &dataInfo,
                                const popops::SlicePlan &plan,
                                unsigned int axis,
                                unsigned int group_size,
                                bool broadcasted,
                                const poplar::DebugNameAndId &dnai);

poplar::Tensor createUpdateTensor(poplar::Graph &graph,
                                  const popart::TensorInfo &dataInfo,
                                  const popart::TensorInfo &indicesInfo,
                                  const popops::SlicePlan &plan,
                                  unsigned int axis,
                                  unsigned int group_size,
                                  bool broadcasted,
                                  const poplar::DebugNameAndId &dnai);

poplar::Tensor createIndicesTensor(poplar::Graph &graph,
                                   const popart::TensorInfo &indicesInfo,
                                   const popops::SlicePlan &plan,
                                   unsigned int axis,
                                   unsigned int group_size,
                                   bool broadcasted,
                                   const poplar::DebugNameAndId &dnai);

// Align input to have same axes alignment and shape as popart IR.
poplar::Tensor alignToAxis(const poplar::Tensor &input,
                           const popart::Shape &shape,
                           unsigned int axis,
                           unsigned int group_size);
} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SLICEPLANX_HPP_
