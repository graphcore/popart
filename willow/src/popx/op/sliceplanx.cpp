// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/sliceplanx.hpp>

#include <popops/DynamicSlice.hpp>

namespace popart {
namespace popx {

popops::SlicePlan
createSlicePlan(const snap::Graph &graph,
                const popart::TensorInfo &dataInfo,
                const popart::TensorInfo &indicesInfo,
                nonstd::optional<float> availableMemoryProportion,
                nonstd::optional<size_t> axis) {
  auto numEntries   = static_cast<size_t>(dataInfo.nelms());
  auto numLookups   = static_cast<size_t>(indicesInfo.nelms());
  size_t outputSize = 1;

  if (numLookups == 0 || numEntries == 0) {
    return popops::SlicePlan();
  }

  if (axis.has_value()) {
    numEntries = dataInfo.shape_szt()[*axis];
    outputSize = dataInfo.nelms() / numEntries;
  }

  // TODO(T40999): this can be removed once "cycles" is made the default
  poplar::OptionFlags opts{{"planMinimisationTarget", "cycles"}};

  if (availableMemoryProportion.has_value()) {
    opts.set("availableMemoryProportion",
             std::to_string(*availableMemoryProportion));
  }

  return popops::embedding::plan(graph.getPoplarGraph(),
                                 popType(dataInfo),
                                 numEntries,
                                 outputSize,
                                 {numLookups},
                                 opts);
}

} // namespace popx
} // namespace popart