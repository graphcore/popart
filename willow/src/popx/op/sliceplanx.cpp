// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/sliceplanx.hpp>

#include <popops/DynamicSlice.hpp>

namespace popart {
namespace popx {

popops::SlicePlan createSlicePlan(const poplar::Graph &graph,
                                  const popart::TensorInfo &dataInfo,
                                  const popart::TensorInfo &indicesInfo) {
  auto numEntries   = static_cast<size_t>(dataInfo.nelms());
  auto numLookups   = static_cast<size_t>(indicesInfo.nelms());
  size_t outputSize = 1;

  if (numLookups == 0) {
    // no lookups, no plan
    return popops::SlicePlan();
  }

  return popops::embedding::plan(
      graph, popType(dataInfo), numEntries, outputSize, {numLookups}, {});
}

} // namespace popx
} // namespace popart