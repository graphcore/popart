// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <ext/new_allocator.h>
#include <string>
#include <vector>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <popops/DynamicSlice.hpp>
#include <poprithms/ndarray/accessors.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/sliceplanx.hpp>

#include "popart/error.hpp"
#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/optional.hpp"

namespace poplar {
class Graph;
} // namespace poplar

namespace nd = poprithms::ndarray;

namespace popart {
namespace popx {

poplar::OptionFlags
createSlicePlanOptions(SlicePlanUsedFor usedFor,
                       nonstd::optional<float> availableMemoryProportion) {
  poplar::OptionFlags opts{};

  if (availableMemoryProportion.has_value()) {
    opts.set("availableMemoryProportion",
             std::to_string(*availableMemoryProportion));
  }

  switch (usedFor) {
  case SlicePlanUsedFor::Slice:
    opts.set("usedForSlice", "true");
    opts.set("usedForUpdate", "false");
    opts.set("operationForUpdate", "none");
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

  case SlicePlanUsedFor::UpdateMax:
    opts.set("usedForSlice", "false");
    opts.set("usedForUpdate", "true");
    opts.set("operationForUpdate", "max");
    break;

  case SlicePlanUsedFor::CombinedSliceUpdate:
    opts.set("usedForSlice", "true");
    opts.set("usedForUpdate", "true");
    opts.set("operationForUpdate", "none");
    break;

  default:
    throw internal_error("Unhandled SlicePlanUsedFor enum");
  }

  return opts;
}

popops::SlicePlan createSlicePlan(const poplar::Graph &graph,
                                  const popart::TensorInfo &sliceableInfo,
                                  const popart::TensorInfo &indicesInfo,
                                  const poplar::OptionFlags &options,
                                  nonstd::optional<size_t> axis,
                                  size_t group_size) {
  const bool isGrouped = group_size > 1;
  auto numEntries   = static_cast<size_t>(sliceableInfo.nelms()) / group_size;
  auto numLookups   = static_cast<size_t>(indicesInfo.nelms()) / group_size;
  size_t outputSize = 1;

  if (numLookups == 0 || numEntries == 0) {
    return popops::SlicePlan();
  }

  if (axis.has_value()) {
    numEntries = sliceableInfo.shape_szt()[*axis];
    outputSize = (sliceableInfo.nelms() / numEntries) / group_size;
  }

  if (isGrouped)
    return popops::embedding::plan(graph,
                                   popType(sliceableInfo),
                                   group_size,
                                   numEntries,
                                   outputSize,
                                   {numLookups},
                                   options);
  else
    return popops::embedding::plan(graph,
                                   popType(sliceableInfo),
                                   numEntries,
                                   outputSize,
                                   {numLookups},
                                   options);
}

poplar::Tensor createDataTensor(poplar::Graph &graph,
                                const popart::TensorInfo &dataInfo,
                                const popops::SlicePlan &plan,
                                unsigned int axis,
                                unsigned int group_size,
                                bool broadcasted,
                                const poplar::DebugNameAndId &dnai) {
  auto numSlices  = static_cast<size_t>(dataInfo.nelms()) / group_size;
  auto outputSize = 1UL;

  if (!broadcasted) {
    numSlices  = dataInfo.shape_szt().at(axis);
    outputSize = (dataInfo.nelms() / numSlices) / group_size;
  }

  const bool isGrouped = group_size > 1;
  auto out             = isGrouped
                 ? popops::createGroupedSliceableTensor(graph,
                                                        popType(dataInfo),
                                                        group_size,
                                                        {numSlices, outputSize},
                                                        {0},
                                                        {1},
                                                        plan,
                                                        poplar::OptionFlags(),
                                                        dnai)
                 : popops::createSliceableTensor(graph,
                                                 popType(dataInfo),
                                                 {numSlices, outputSize},
                                                 {0},
                                                 {1},
                                                 plan,
                                                 poplar::OptionFlags(),
                                                 dnai);

  return alignToAxis(poplar::Tensor{out}, dataInfo.shape(), axis, group_size);
}

poplar::Tensor createUpdateTensor(poplar::Graph &graph,
                                  const popart::TensorInfo &dataInfo,
                                  const popart::TensorInfo &indicesInfo,
                                  const popops::SlicePlan &plan,
                                  unsigned int axis,
                                  unsigned int group_size,
                                  bool broadcasted,
                                  const poplar::DebugNameAndId &dnai) {
  auto numSlices  = static_cast<size_t>(dataInfo.nelms()) / group_size;
  auto outputSize = 1UL;

  if (!broadcasted) {
    numSlices  = dataInfo.shape_szt().at(axis);
    outputSize = (dataInfo.nelms() / numSlices) / group_size;
  }

  auto numLookups      = static_cast<size_t>(indicesInfo.nelms()) / group_size;
  const bool isGrouped = group_size > 1;
  auto out             = isGrouped
                 ? popops::createGroupedSliceTensor(graph,
                                                    popType(dataInfo),
                                                    group_size,
                                                    {numSlices, outputSize},
                                                    {0},
                                                    {1},
                                                    numLookups,
                                                    plan,
                                                    poplar::OptionFlags(),
                                                    dnai)
                 : popops::createSliceTensor(graph,
                                             popType(dataInfo),
                                             {numSlices, outputSize},
                                             {0},
                                             {1},
                                             numLookups,
                                             plan,
                                             poplar::OptionFlags(),
                                             dnai);
  if (broadcasted) {
    return alignToAxis(
        poplar::Tensor{out}, indicesInfo.shape(), axis, group_size);
  }

  auto shape  = dataInfo.shape();
  shape[axis] = numLookups;
  return alignToAxis(poplar::Tensor{out}, shape, axis, group_size);
}

poplar::Tensor createIndicesTensor(poplar::Graph &graph,
                                   const popart::TensorInfo &indicesInfo,
                                   const popops::SlicePlan &plan,
                                   unsigned int axis,
                                   unsigned int group_size,
                                   bool broadcasted,
                                   const poplar::DebugNameAndId &dnai) {
  auto numLookups      = static_cast<size_t>(indicesInfo.nelms()) / group_size;
  const bool isGrouped = group_size > 1;
  auto indices =
      isGrouped
          ? popops::createGroupedIndicesTensor(graph,
                                               group_size,
                                               {0},
                                               numLookups,
                                               plan,
                                               poplar::OptionFlags(),
                                               dnai)
          : popops::createIndicesTensor(
                graph, {0}, numLookups, plan, poplar::OptionFlags(), dnai);

  indices = indices.reinterpret(popType(indicesInfo));

  if (broadcasted) {
    return alignToAxis(
        poplar::Tensor{indices}, indicesInfo.shape(), axis, group_size);
  }

  indices = indices.reshape(indicesInfo.shape_szt());
  return indices;
}

poplar::Tensor alignToAxis(const poplar::Tensor &input,
                           const popart::Shape &shape,
                           unsigned axis,
                           unsigned group_size) {
  const unsigned axisStartDim = group_size > 1 ? 1 : 0;
  nd::Shape ndShape(shape);
  ndShape = ndShape.dimRoll(nd::Dimension(axis), nd::Dimension(axisStartDim));
  auto output = input.reshape(ndShape.get_u64());
  output      = output.dimRoll(axisStartDim, axis);
  return output;
}

} // namespace popx
} // namespace popart
