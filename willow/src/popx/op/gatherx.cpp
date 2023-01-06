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
  axis = dynamic_cast<const GatherOp *>(op)->getAxis();

  // We always want the gather to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

GatherOpx::GatherOpx(Op *op, Devicex *devicex) : GatherBaseOpx(op, devicex) {
  verifyOp<GatherOp>(op,
                     {Onnx::Operators::Gather_1, Onnx::Operators::Gather_11});

  setCommonMembersPostVerify(op);

  const auto &gop = getOp<GatherOp>();
  auto options    = createSlicePlanOptions(SlicePlanUsedFor::Slice,
                                        gop.getAvailableMemoryProportion());
  plan            = createSlicePlan(graph(),
                         inInfo(gop.dataInIndex()),
                         inInfo(gop.indicesInIndex()),
                         options,
                         axis);
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
  offsets = offsets.expand({1});

  // Place the gather axis at the front.
  data = data.dimRoll(static_cast<unsigned>(axis));
  // Store the shape for later.
  auto tmp_shape = data.shape();
  // Flatten the other dimensions.
  data = data.flatten(1, data.rank());

  poplar::Tensor mask;
  if (op.zeroOutOfRangeIndices()) {
    std::tie(offsets, mask) =
        zeroIndiciesThatAreOutOfRange(prog, data, offsets);
  }

  offsets = offsets.reinterpret(poplar::UNSIGNED_INT);

  auto result = popops::multiSlice(graph(),
                                   data,
                                   offsets,
                                   {0},
                                   {1},
                                   prog,
                                   plan,
                                   poplar::OptionFlags(),
                                   debugContext());

  if (op.zeroOutOfRangeIndices()) {
    zeroOutputOfOutOfRangeIndices(prog, result, mask, data);
  }

  // Reshape the result to "unflatten" the other dimensions.
  tmp_shape.front() = result.dim(0);
  result            = result.reshape(tmp_shape);
  // Put the gather axis dimension back in the right place.
  result = result.dimRoll(0, static_cast<unsigned>(axis));

  // Reshape into the expected ONNX shape.
  result = result.reshape(outputShape);

  setOutTensor(GatherOp::outIndex(), result);
}

std::tuple<poplar::Tensor, poplar::Tensor>
GatherBaseOpx::zeroIndiciesThatAreOutOfRange(
    poplar::program::Sequence &prog,
    const poplar::Tensor &data,
    const poplar::Tensor &offsets) const {
  auto gather_size = data.shape()[0];
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
  std::vector<size_t> dims  = {static_cast<size_t>(axis)};
  std::vector<size_t> sizes = {1};

  if (index == GatherOp::dataInIndex()) {
    auto dataInfo        = inInfo(index);
    const auto dataShape = dataInfo.shape_szt();
    auto data            = popops::createSliceableTensor(graph(),
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
  auto indices     = popops::createIndicesTensor(
      graph(), dims, indicesInfo.nelms(), plan, poplar::OptionFlags(), dnai);
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
  verifyOp<GatherGradOp>(op, Onnx::GradOperators::GatherGrad);

  auto &gop    = getOp<GatherGradOp>();
  axis         = gop.getAxis();
  auto options = createSlicePlanOptions(SlicePlanUsedFor::UpdateAdd,
                                        gop.getAvailableMemoryProportion());
  plan         = createSlicePlan(graph(),
                         outInfo(gop.gradOutIndex()),
                         inInfo(gop.indicesInIndex()),
                         options,
                         axis);

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
  std::vector<size_t> dims  = {static_cast<size_t>(axis)};
  std::vector<size_t> sizes = {1};

  if (index == GatherGradOp::gradInIndex()) {
    auto gradInfo        = inInfo(index);
    const auto dataShape = gradInfo.shape_szt();

    return popops::createSliceableTensor(graph(),
                                         popType(gradInfo),
                                         dataShape,
                                         dims,
                                         sizes,
                                         plan,
                                         poplar::OptionFlags(),
                                         dnai);
  }

  auto indicesInfo = inInfo(index);
  auto indices     = popops::createIndicesTensor(
      graph(), dims, indicesInfo.nelms(), plan, poplar::OptionFlags(), dnai);
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
                                   int64_t axis) {
  // Flatten the index shaped region of the update
  update = update.flatten(static_cast<unsigned>(axis),
                          static_cast<unsigned>(axis) + indices.rank());
  // Put the slice dimension at the front
  update = update.dimRoll(static_cast<unsigned>(axis));
  // Flatten the rest of the dimensions
  update = update.flatten(1, update.rank());
  // Add a degenerate dimension
  update = update.expand({1});

  // Put the slice dimension at the front
  target = target.dimRoll(static_cast<unsigned>(axis));
  // Flatten the rest of the dimensions
  target = target.flatten(1, target.rank());

  // Flatten the indices to a vector
  indices = indices.flatten();
  // Add a degenerate dimension
  indices = indices.expand({1});
  // Reinterpret the indices as unsigned int, assuming negative indices don't
  // exist.
  indices = indices.reinterpret(poplar::UNSIGNED_INT);

  return {target, update, indices};
}

void GatherGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(GatherGradOp::gradOutIndex()));

  auto update  = getInTensor(GatherGradOp::gradInIndex());
  auto indices = getInTensor(GatherGradOp::indicesInIndex());

  auto result = popops::createSliceableTensor(graph(),
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
  const auto inputs = handleNDMultiUpdate(result, update, indices, axis);
  auto &targetND    = std::get<0>(inputs);
  auto &updateND    = std::get<1>(inputs);
  auto &indicesND   = std::get<2>(inputs);

  // Accumulate the updates into the target
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

  setOutTensor(GatherGradOp::gradOutIndex(), result);
}

namespace {
OpxCreator<GatherOpx> gatherOpxCreator({Onnx::Operators::Gather_1,
                                        Onnx::Operators::Gather_11});
OpxCreator<GatherGradOpx> gatherGradOpxCreator(Onnx::GradOperators::GatherGrad);
} // namespace

} // namespace popx
} // namespace popart
