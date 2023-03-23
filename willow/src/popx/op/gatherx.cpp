// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ext/new_allocator.h>
#include <limits>
#include <set>
#include <tuple>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>
#include <popart/error.hpp>
#include <popart/op/gather.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/gatherx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
namespace popx {
class Devicex;

GatherBaseOpx::GatherBaseOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

void GatherBaseOpx::setCommonMembersPostVerify(const Op *op) {
  // Note TiedGatherOp extends GatherOp.

  axis       = dynamic_cast<const GatherOp *>(op)->getAxis();
  group_size = dynamic_cast<const GatherOp *>(op)->getGroupSize();

  // We always want the gather to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

GatherOpx::GatherOpx(Op *op, Devicex *devicex) : GatherBaseOpx(op, devicex) {
  verifyOp<GatherOp>(op,
                     {Onnx::Operators::Gather_1,
                      Onnx::Operators::Gather_11,
                      Onnx::CustomOperators::GroupedGather});

  setCommonMembersPostVerify(op);

  const auto &gop        = getOp<GatherOp>();
  auto options           = createSlicePlanOptions(SlicePlanUsedFor::Slice,
                                        gop.getAvailableMemoryProportion());
  const auto dataInfo    = inInfo(gop.dataInIndex());
  const auto indicesInfo = inInfo(gop.indicesInIndex());
  plan                   = createSlicePlan(
      graph(), dataInfo, indicesInfo, options, axis, group_size);

  // Check that indices or data are not empty tensors. If yes, set the group
  // size to 1 because grouped verison require a SlicePlan which cannot be
  // defined in this situation.
  if (indicesInfo.nelms() == 0 || dataInfo.nelms() == 0)
    group_size = 1;
}

void GatherOpx::grow(poplar::program::Sequence &prog) const {
  const auto outputShape = outInfo(GatherOp::outIndex()).shape_szt();
  auto indices           = getInTensor(GatherOp::indicesInIndex());
  auto data              = getInTensor(GatherOp::dataInIndex());

  const auto &op = getOp<GatherOp>();

  // If there are no indices, return an empty tensor of the appropriate
  // shape
  if (indices.numElements() == 0) {
    auto result = graph().addVariable(
        data.elementType(), outputShape, debugContext("result"));

    setOutTensor(GatherOp::outIndex(), result);
    return;
  }

  // Flatten the scalar indices.
  auto offsets = indices.flatten();
  // Add a degenerate dimension at the end.
  offsets              = offsets.expand({1});
  unsigned ugroup_size = static_cast<unsigned>(group_size);

  if (isGrouped())
    offsets = offsets.reshapePartial(
        0, 1, {ugroup_size, offsets.dim(0) / ugroup_size});

  // Place the gather axis at the front.
  data = data.dimRoll(static_cast<unsigned>(axis), isGrouped() ? 1 : 0);
  // Store the shape for later.
  auto tmpShape = data.shape();
  // Flatten the other dimensions.
  data = data.flatten(isGrouped() ? 2 : 1, data.rank());

  poplar::Tensor mask;
  if (op.zeroOutOfRangeIndices()) {
    std::tie(offsets, mask) =
        zeroIndiciesThatAreOutOfRange(prog, data, offsets);
  }

  offsets = offsets.reinterpret(poplar::UNSIGNED_INT);
  auto result =
      isGrouped()
          ? popops::groupedMultiSlice(graph(),
                                      data,
                                      offsets,
                                      {0},
                                      {1},
                                      prog,
                                      plan,
                                      poplar::OptionFlags(),
                                      debugContext("GroupedGatherResult"))
          : popops::multiSlice(graph(),
                               data,
                               offsets,
                               {0},
                               {1},
                               prog,
                               plan,
                               poplar::OptionFlags(),
                               debugContext("gatherResult"));

  if (op.zeroOutOfRangeIndices()) {
    zeroOutputOfOutOfRangeIndices(prog, result, mask, data);
  }

  // Reshape the result to "unflatten" the other dimensions.
  tmpShape.front() = result.dim(0);
  if (isGrouped())
    tmpShape[1] = result.dim(1);

  result = result.reshape(tmpShape);

  // Put the gather axis dimension back in the right place.
  result = result.dimRoll(isGrouped() ? 1 : 0, static_cast<unsigned>(axis));

  // Reshape into the expected ONNX shape.
  result = result.reshape(outputShape);

  if (isGrouped()) {
    const poplar::Tensor remapped_result = graph().addVariable(
        result.elementType(), result.shape(), "RemappedResult");
    for (int g = 0; g < group_size; g++)
      poputil::mapTensorLinearly(graph(), remapped_result.slice(g, g + 1, 0));
    prog.add(poplar::program::Copy(result, remapped_result));
    setOutTensor(GatherOp::outIndex(), remapped_result);
  } else {
    setOutTensor(GatherOp::outIndex(), result);
  }
}

std::tuple<poplar::Tensor, poplar::Tensor>
GatherBaseOpx::zeroIndiciesThatAreOutOfRange(
    poplar::program::Sequence &prog,
    const poplar::Tensor &data,
    const poplar::Tensor &offsets) const {
  auto gather_size = data.shape()[isGrouped() ? 1 : 0];
  auto dtype       = offsets.elementType();
  auto max_value   = getConst(dtype, {}, gather_size, "max_value");
  auto mask =
      popops::lt(graph(), offsets, max_value, prog, debugContext("mask<size"));

  if (dtype == poplar::INT || dtype == poplar::SHORT || dtype == poplar::LONG ||
      dtype == poplar::LONGLONG) {
    auto mask2 =
        popops::gteq(graph(), offsets, 0, prog, debugContext("0<mask"));
    popops::logicalAndInPlace(
        graph(), mask, mask2, prog, debugContext("0<=mask<size"));
  }
  auto indices_mask = popops::cast(
      graph(), mask, offsets.elementType(), prog, debugContext("mask_castInt"));
  auto masked_offsets = popops::mul(
      graph(), offsets, indices_mask, prog, debugContext("masked_indices"));
  return std::make_tuple(masked_offsets, mask);
}

void GatherBaseOpx::zeroOutputOfOutOfRangeIndices(
    poplar::program::Sequence &prog,
    poplar::Tensor &result,
    const poplar::Tensor &mask,
    const poplar::Tensor &data) const {
  auto out_mask = popops::cast(
      graph(), mask, data.elementType(), prog, debugContext("mask_cast"));
  popops::mulInPlace(graph(),
                     result,
                     out_mask.expand({1}),
                     prog,
                     debugContext("masked_result"));
}

poplar::Tensor
GatherOpx::createInput(InIndex index,
                       const poplar::DebugNameAndId &dnai) const {
  if (index != GatherOp::dataInIndex() && index != GatherOp::indicesInIndex()) {
    throw error("GatherOpx::createInput Cannot create input {}", index);
  }

  std::vector<size_t> dims  = {static_cast<size_t>(axis) -
                              static_cast<size_t>(isGrouped())};
  std::vector<size_t> sizes = {1};

  if (index == GatherOp::dataInIndex()) {
    auto dataInfo  = inInfo(index);
    auto dataShape = dataInfo.shape_szt();
    if (isGrouped())
      dataShape.erase(dataShape.begin());

    auto data =
        isGrouped()
            ? popops::createGroupedSliceableTensor(graph(),
                                                   popType(dataInfo),
                                                   group_size,
                                                   dataShape,
                                                   dims,
                                                   sizes,
                                                   plan,
                                                   poplar::OptionFlags(),
                                                   dnai)
            : popops::createSliceableTensor(graph(),
                                            popType(dataInfo),
                                            dataShape,
                                            dims,
                                            sizes,
                                            plan,
                                            poplar::OptionFlags(),
                                            dnai);

    return data;
  }
  auto indicesInfo = inInfo(index);
  auto numLookups  = static_cast<size_t>(indicesInfo.nelms()) / group_size;
  auto indices =
      isGrouped()
          ? popops::createGroupedIndicesTensor(graph(),
                                               group_size,
                                               dims,
                                               numLookups,
                                               plan,
                                               poplar::OptionFlags(),
                                               dnai)
          : popops::createIndicesTensor(
                graph(), dims, numLookups, plan, poplar::OptionFlags(), dnai);
  indices = indices.reinterpret(popType(indicesInfo));
  indices = indices.reshape(indicesInfo.shape_szt());
  return indices;
}

InputCreatorType GatherOpx::getInputCreatorType(InIndex index) const {
  if (inInfo(index).nelms() == 0) {
    return Opx::getInputCreatorType(index);
  }

  if (index == GatherOp::dataInIndex() || index == GatherOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return Opx::getInputCreatorType(index);
}

std::set<TensorId> GatherBaseOpx::mustExistBeforeCreate(int) const {
  return {};
}

GatherGradOpx::GatherGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GatherGradOp>(op,
                         {Onnx::GradOperators::GatherGrad,
                          Onnx::CustomGradOperators::GroupedGatherGrad});

  auto &gop              = getOp<GatherGradOp>();
  axis                   = gop.getAxis();
  group_size             = gop.getGroupSize();
  auto options           = createSlicePlanOptions(SlicePlanUsedFor::UpdateAdd,
                                        gop.getAvailableMemoryProportion());
  const auto indicesInfo = inInfo(gop.indicesInIndex());
  const auto outputInfo  = outInfo(gop.gradOutIndex());
  plan                   = createSlicePlan(
      graph(), outputInfo, indicesInfo, options, axis, group_size);
  // Check that indices or data are not empty tensors. If yes, set the group
  // size to 1 because grouped verison require a SlicePlan which cannot be
  // defined in this situation.
  if (indicesInfo.nelms() == 0 || outputInfo.nelms() == 0)
    group_size = 1;

  // We always want this op to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

poplar::Tensor
GatherGradOpx::createInput(InIndex index,
                           const poplar::DebugNameAndId &dnai) const {
  if (index != GatherGradOp::gradInIndex() &&
      index != GatherGradOp::indicesInIndex()) {
    throw error("GatherGradOpx::createInput Cannot create input {}", index);
  }
  std::vector<size_t> dims  = {static_cast<size_t>(axis) -
                              static_cast<size_t>(isGrouped())};
  std::vector<size_t> sizes = {1};

  if (index == GatherGradOp::gradInIndex()) {
    auto gradInfo  = inInfo(index);
    auto dataShape = gradInfo.shape_szt();
    if (isGrouped())
      dataShape.erase(dataShape.begin());

    return isGrouped()
               ? popops::createGroupedSliceableTensor(graph(),
                                                      popType(gradInfo),
                                                      group_size,
                                                      dataShape,
                                                      dims,
                                                      sizes,
                                                      plan,
                                                      poplar::OptionFlags(),
                                                      dnai)
               : popops::createSliceableTensor(graph(),
                                               popType(gradInfo),
                                               dataShape,
                                               dims,
                                               sizes,
                                               plan,
                                               poplar::OptionFlags(),
                                               dnai);
  }

  auto indicesInfo = inInfo(index);
  auto numLookups  = static_cast<size_t>(indicesInfo.nelms()) / group_size;
  auto indices =
      group_size
          ? popops::createGroupedIndicesTensor(graph(),
                                               group_size,
                                               dims,
                                               numLookups,
                                               plan,
                                               poplar::OptionFlags(),
                                               dnai)
          : popops::createIndicesTensor(
                graph(), dims, numLookups, plan, poplar::OptionFlags(), dnai);
  indices = indices.reinterpret(popType(indicesInfo));
  return indices.reshape(indicesInfo.shape_szt());
}

InputCreatorType GatherGradOpx::getInputCreatorType(InIndex index) const {
  if (inInfo(index).nelms() == 0) {
    return Opx::getInputCreatorType(index);
  }

  if (index == GatherGradOp::gradInIndex() ||
      index == GatherGradOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return Opx::getInputCreatorType(index);
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
GatherGradOpx::handleNDMultiUpdate(poplar::Tensor target,
                                   poplar::Tensor update,
                                   poplar::Tensor indices,
                                   int64_t axis,
                                   int64_t group_size) {
  bool isGrouped = group_size > 1;
  // Flatten the index shaped region of the update
  const unsigned flattenRange =
      indices.rank() - static_cast<unsigned>(isGrouped);
  update = update.flatten(static_cast<unsigned>(axis),
                          static_cast<unsigned>(axis) + flattenRange);
  // Put the slice dimension at the front
  update = update.dimRoll(static_cast<unsigned>(axis), isGrouped ? 1 : 0);
  // Flatten the rest of the dimensions
  update = update.flatten(isGrouped ? 2 : 1, update.rank());
  // Add a degenerate dimension
  update = update.expand({isGrouped ? 2U : 1U});

  // Put the slice dimension at the front
  target = target.dimRoll(static_cast<unsigned>(axis), isGrouped ? 1 : 0);
  // Flatten the rest of the dimensions
  target = target.flatten(isGrouped ? 2 : 1, target.rank());

  // Flatten the indices to a vector
  indices = indices.flatten();
  // Add a degenerate dimension
  indices = indices.expand({1});
  if (isGrouped) {
    unsigned ugroup_size = static_cast<unsigned>(group_size);
    indices              = indices.reshapePartial(
        0, 1, {ugroup_size, indices.dim(0) / ugroup_size});
  }
  // Reinterpret the indices as unsigned int, assuming negative indices don't
  // exist.
  indices = indices.reinterpret(poplar::UNSIGNED_INT);

  return {target, update, indices};
}

void GatherGradOpx::grow(poplar::program::Sequence &prog) const {
  auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(GatherGradOp::gradOutIndex()));

  auto update  = getInTensor(GatherGradOp::gradInIndex());
  auto indices = getInTensor(GatherGradOp::indicesInIndex());

  if (isGrouped())
    outputShape.erase(outputShape.begin());

  auto result =
      isGrouped()
          ? popops::createGroupedSliceableTensor(
                graph(),
                update.elementType(),
                group_size,
                outputShape,
                {static_cast<size_t>(axis - 1)},
                {1},
                plan,
                poplar::OptionFlags(),
                debugContext("groupedGatherGradResult"))

          : popops::createSliceableTensor(graph(),
                                          update.elementType(),
                                          outputShape,
                                          {static_cast<size_t>(axis)},
                                          {1},
                                          plan,
                                          poplar::OptionFlags(),
                                          debugContext("gatherGradResult"));

  // Zero the result tensor
  popops::zero(graph(), result, prog, debugContext("zero"));

  if (result.numElements() == 0 || update.numElements() == 0 ||
      indices.numElements() == 0) {
    setOutTensor(GatherGradOp::gradOutIndex(), result);
    return;
  }

  auto scale = graph().addConstant(
      update.elementType(), {}, 1.0f, debugContext("const_1"));
  graph().setTileMapping(scale, 0);
  // Rolls axis to front.
  const auto inputs =
      handleNDMultiUpdate(result, update, indices, axis, group_size);
  auto &targetND  = std::get<0>(inputs);
  auto &updateND  = std::get<1>(inputs);
  auto &indicesND = std::get<2>(inputs);

  // Accumulate the updates into the target
  if (isGrouped()) {
    popops::groupedMultiUpdateAdd(graph(),
                                  targetND,
                                  updateND,
                                  indicesND,
                                  scale,
                                  {0},
                                  {1},
                                  prog,
                                  plan,
                                  poplar::OptionFlags(),
                                  debugContext("groupedGatherGrad"));
  } else {
    popops::multiUpdateAdd(graph(),
                           targetND,
                           updateND,
                           indicesND,
                           scale,
                           {0},
                           {1},
                           prog,
                           plan,
                           poplar::OptionFlags(),
                           debugContext("gatherGrad"));
  }
  if (isGrouped()) {
    const poplar::Tensor remapped_result = graph().addVariable(
        result.elementType(), result.shape(), "RemappedGradOut");
    for (int g = 0; g < group_size; g++)
      poputil::mapTensorLinearly(graph(), remapped_result.slice(g, g + 1, 0));
    prog.add(poplar::program::Copy(result, remapped_result));
    setOutTensor(GatherGradOp::gradOutIndex(), remapped_result);
  } else {
    setOutTensor(GatherGradOp::gradOutIndex(), result);
  }
}

namespace {
OpxCreator<GatherOpx> gatherOpxCreator({Onnx::Operators::Gather_1,
                                        Onnx::Operators::Gather_11,
                                        Onnx::CustomOperators::GroupedGather});
OpxCreator<GatherGradOpx>
    gatherGradOpxCreator({Onnx::GradOperators::GatherGrad,
                          Onnx::CustomGradOperators::GroupedGatherGrad});
} // namespace

} // namespace popx
} // namespace popart
