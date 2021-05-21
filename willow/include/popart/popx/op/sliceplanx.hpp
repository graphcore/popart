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

popops::SlicePlan createSlicePlan(const snap::Graph &graph,
                                  const popart::TensorInfo &dataInfo,
                                  const popart::TensorInfo &indicesInfo,
                                  nonstd::optional<size_t> axis = {});

} // namespace popx
} // namespace popart

#endif
