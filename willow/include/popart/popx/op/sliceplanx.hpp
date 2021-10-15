// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SLICEPLANX_HPP
#define GUARD_NEURALNET_SLICEPLANX_HPP

#include <popart/names.hpp>
#include <popart/vendored/optional.hpp>

#include <popops/DynamicSlice.hpp>

namespace poplar {
class Graph;
}

namespace popart {
namespace popx {

enum class SlicePlanUsedFor { Slice, Update, UpdateAdd };

poplar::OptionFlags
createSlicePlanOptions(SlicePlanUsedFor usedFor,
                       nonstd::optional<float> availableMemoryProportion = {});

popops::SlicePlan createSlicePlan(const snap::Graph &graph,
                                  const popart::TensorInfo &dataInfo,
                                  const popart::TensorInfo &indicesInfo,
                                  const poplar::OptionFlags &options,
                                  nonstd::optional<size_t> axis = {});

// Align input to have same axes alignment and shape as popart IR.
snap::Tensor alignToAxis(const snap::Tensor &input,
                         const popart::Shape &shape,
                         unsigned int axis);
} // namespace popx
} // namespace popart

#endif
