// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SLICEPLANX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SLICEPLANX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <cstddef>
#include <snap/Tensor.hpp>
#include <poplar/OptionFlags.hpp>
#include <popops/DynamicSlice.hpp>
#include <popart/names.hpp>
#include <popart/vendored/optional.hpp>

namespace snap {
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

popops::SlicePlan createSlicePlan(const snap::Graph &graph,
                                  const popart::TensorInfo &sliceableInfo,
                                  const popart::TensorInfo &indicesInfo,
                                  const poplar::OptionFlags &options,
                                  nonstd::optional<size_t> axis = {});

snap::Tensor createDataTensor(snap::Graph &graph,
                              const popart::TensorInfo &dataInfo,
                              const popops::SlicePlan &plan,
                              unsigned int axis,
                              bool broadcasted,
                              const poplar::DebugNameAndId &dnai);

snap::Tensor createUpdateTensor(snap::Graph &graph,
                                const popart::TensorInfo &dataInfo,
                                const popart::TensorInfo &indicesInfo,
                                const popops::SlicePlan &plan,
                                unsigned int axis,
                                bool broadcasted,
                                const poplar::DebugNameAndId &dnai);

snap::Tensor createIndicesTensor(snap::Graph &graph,
                                 const popart::TensorInfo &indicesInfo,
                                 const popops::SlicePlan &plan,
                                 unsigned int axis,
                                 bool broadcasted,
                                 const poplar::DebugNameAndId &dnai);

// Align input to have same axes alignment and shape as popart IR.
snap::Tensor alignToAxis(const snap::Tensor &input,
                         const popart::Shape &shape,
                         unsigned int axis);
} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SLICEPLANX_HPP_
