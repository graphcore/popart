// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/ndarray/shape.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/sliceplanx.hpp>

#include <popops/DynamicSlice.hpp>

namespace nd = poprithms::ndarray;

namespace popart {
namespace popx {

poplar::OptionFlags
createSlicePlanOptions(SlicePlanUsedFor usedFor,
                       nonstd::optional<float> availableMemoryProportion) {
  // TODO(T40999): this can be removed once "cycles" is made the default
  poplar::OptionFlags opts{};

  if (availableMemoryProportion.has_value()) {
    opts.set("availableMemoryProportion",
             std::to_string(*availableMemoryProportion));
  }

  switch (usedFor) {
  case SlicePlanUsedFor::Slice:
    opts.set("usedForSlice", "true");
    opts.set("usedForUpdate", "false");
    break;

  case SlicePlanUsedFor::Update:
    opts.set("usedForSlice", "false");
    opts.set("usedForUpdate", "true");
    opts.set("operationForUpdate", "none");
    break;

  case SlicePlanUsedFor::UpdateAdd:
    opts.set("usedForSlice", "false");
    opts.set("usedForUpdate", "true");
    opts.set("operationForUpdate", "add");
    break;

  default:
    throw internal_error("Unhandled SlicePlanUsedFor enum");
  }

  return opts;
}

popops::SlicePlan createSlicePlan(const snap::Graph &graph,
                                  const popart::TensorInfo &sliceableInfo,
                                  const popart::TensorInfo &indicesInfo,
                                  const poplar::OptionFlags &options,
                                  nonstd::optional<size_t> axis) {
  auto numEntries   = static_cast<size_t>(sliceableInfo.nelms());
  auto numLookups   = static_cast<size_t>(indicesInfo.nelms());
  size_t outputSize = 1;

  if (numLookups == 0 || numEntries == 0) {
    return popops::SlicePlan();
  }

  if (axis.has_value()) {
    numEntries = sliceableInfo.shape_szt()[*axis];
    outputSize = sliceableInfo.nelms() / numEntries;
  }

  return popops::embedding::plan(graph.getPoplarGraph(),
                                 popType(sliceableInfo),
                                 numEntries,
                                 outputSize,
                                 {numLookups},
                                 options);
}

snap::Tensor createDataTensor(snap::Graph &graph,
                              const popart::TensorInfo &dataInfo,
                              const popops::SlicePlan &plan,
                              unsigned int axis,
                              const poplar::DebugNameAndId &dnai) {
  auto numEntries = static_cast<size_t>(dataInfo.nelms());
  auto out        = popops::createSliceableTensor(graph.getPoplarGraph(),
                                           popType(dataInfo),
                                           {numEntries, 1},
                                           {0},
                                           {1},
                                           plan,
                                           poplar::OptionFlags(),
                                           dnai);

  return alignToAxis(snap::Tensor{out, graph}, dataInfo.shape(), axis);
}

snap::Tensor createUpdateTensor(snap::Graph &graph,
                                const popart::TensorInfo &dataInfo,
                                const popart::TensorInfo &indicesInfo,
                                const popops::SlicePlan &plan,
                                unsigned int axis,
                                const poplar::DebugNameAndId &dnai) {
  auto numSlices  = static_cast<size_t>(dataInfo.nelms());
  auto numLookups = static_cast<size_t>(indicesInfo.nelms());
  auto out        = popops::createSliceTensor(graph.getPoplarGraph(),
                                       popType(dataInfo),
                                       {numSlices, 1},
                                       {0},
                                       {1},
                                       numLookups,
                                       plan,
                                       poplar::OptionFlags(),
                                       dnai);

  return alignToAxis(snap::Tensor{out, graph}, indicesInfo.shape(), axis);
}

snap::Tensor createIndicesTensor(snap::Graph &graph,
                                 const popart::TensorInfo &indicesInfo,
                                 const popops::SlicePlan &plan,
                                 unsigned int axis,
                                 const poplar::DebugNameAndId &dnai) {
  auto numLookups = static_cast<size_t>(indicesInfo.nelms());
  auto indices    = popops::createIndicesTensor(graph.getPoplarGraph(),
                                             {0},
                                             numLookups,
                                             plan,
                                             poplar::OptionFlags(),
                                             dnai);

  indices = indices.reinterpret(popType(indicesInfo));
  return alignToAxis(snap::Tensor{indices, graph}, indicesInfo.shape(), axis);
}

snap::Tensor alignToAxis(const snap::Tensor &input,
                         const popart::Shape &shape,
                         unsigned axis) {
  nd::Shape ndShape(shape);
  ndShape     = ndShape.dimRoll(nd::Dimension(axis), nd::Dimension(0));
  auto output = input.reshape(ndShape.get_u64());
  output      = output.dimRoll(0, axis);
  return output;
}

} // namespace popx
} // namespace popart