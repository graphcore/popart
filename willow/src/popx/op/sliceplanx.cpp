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
  poplar::OptionFlags opts{{"planMinimisationTarget", "cycles"}};

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
                                  const popart::TensorInfo &dataInfo,
                                  const popart::TensorInfo &indicesInfo,
                                  const popart::TensorInfo &outputInfo,
                                  const poplar::OptionFlags &options,
                                  nonstd::optional<size_t> axis) {
  auto numEntries = static_cast<size_t>(dataInfo.nelms());
  auto numLookups = static_cast<size_t>(indicesInfo.nelms());
  auto outputSize = static_cast<size_t>(outputInfo.nelms());

  if (numLookups == 0 || numEntries == 0) {
    return popops::SlicePlan();
  }

  if (axis.has_value()) {
    numEntries = dataInfo.shape_szt()[*axis];
    outputSize = outputInfo.shape_szt()[*axis];
  }

  return popops::embedding::plan(graph.getPoplarGraph(),
                                 popType(dataInfo),
                                 numEntries,
                                 outputSize,
                                 {numLookups},
                                 options);
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