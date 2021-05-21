// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/gather.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/gatherx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

GatherOpx::GatherOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex), plan(), axis() {
  verifyOp<GatherOp>(op,
                     {Onnx::Operators::Gather_1, Onnx::Operators::Gather_11});
  auto &gop = getOp<GatherOp>();
  axis      = gop.getAxis();
  plan      = createSlicePlan(graph(),
                         gop.inInfo(gop.dataInIndex()),
                         gop.inInfo(gop.indicesInIndex()),
                         static_cast<size_t>(axis));

  // We always want the gather to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void GatherOpx::grow(poplar::program::Sequence &prog) const {
  const auto outputShape = outInfo(GatherOp::outIndex()).shape_szt();
  auto indices           = getInTensor(GatherOp::indicesInIndex());
  auto data              = getInTensor(GatherOp::dataInIndex());

  // If there are no indices, return an empty tensor of the appropriate
  // shape
  if (indices.numElements() == 0) {
    auto result = graph().getPoplarGraph().addVariable(
        data.elementType(), outputShape, debugContext("result"));

    setOutTensor(GatherOp::outIndex(), result);
    return;
  }

  // Flatten the scalar indices.
  auto offsets = indices.flatten();
  // Add a degenerate dimension at the end.
  offsets = offsets.expand({1});
  // reinterpret the indices as unsigned int. This assumes non-negative indices.
  offsets = offsets.reinterpret(poplar::UNSIGNED_INT);

  // Place the gather axis at the front.
  data = data.dimRoll(static_cast<unsigned>(axis));
  // Store the shape for later.
  auto tmp_shape = data.shape();
  // Flatten the other dimensions.
  data = data.flatten(1, data.rank());

  auto result = popops::multiSlice(graph().getPoplarGraph(),
                                   data,
                                   offsets,
                                   {0},
                                   {1},
                                   prog,
                                   plan,
                                   poplar::OptionFlags(),
                                   debugContext());

  // Reshape the result to "unflatten" the other dimensions.
  tmp_shape.front() = result.dim(0);
  result            = result.reshape(tmp_shape);
  // Put the gather axis dimension back in the right place.
  result = result.dimRoll(0, static_cast<unsigned>(axis));

  // Reshape into the expected ONNX shape.
  result = result.reshape(outputShape);

  setOutTensor(GatherOp::outIndex(), result);
}

poplar::Tensor
GatherOpx::createInput(int index, const poplar::DebugNameAndId &dnai) const {
  if (index != GatherOp::dataInIndex() && index != GatherOp::indicesInIndex()) {
    throw error("GatherOpx::createInput Cannot create input {}", index);
  }
  std::vector<size_t> dims  = {static_cast<size_t>(axis)};
  std::vector<size_t> sizes = {1};

  if (index == GatherOp::dataInIndex()) {
    auto dataInfo        = inInfo(index);
    const auto dataShape = dataInfo.shape_szt();

    return popops::createSliceableTensor(graph().getPoplarGraph(),
                                         popType(dataInfo),
                                         dataShape,
                                         dims,
                                         sizes,
                                         plan,
                                         poplar::OptionFlags(),
                                         dnai);
  }

  auto indicesInfo = inInfo(index);
  auto indices     = popops::createIndicesTensor(graph().getPoplarGraph(),
                                             dims,
                                             indicesInfo.nelms(),
                                             plan,
                                             poplar::OptionFlags(),
                                             dnai);
  indices          = indices.reinterpret(popType(indicesInfo));
  return indices.reshape(indicesInfo.shape_szt());
}

InputCreatorType GatherOpx::getInputCreatorType(int index) const {
  if (index == GatherOp::dataInIndex() || index == GatherOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return PopOpx::getInputCreatorType(index);
}

std::set<TensorId> GatherOpx::mustExistBeforeCreate(int) const { return {}; }

GatherGradOpx::GatherGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<GatherGradOp>(op, Onnx::GradOperators::GatherGrad);

  axis = dynamic_cast<GatherGradOp *>(op)->getAxis();
}

void GatherGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(GatherGradOp::gradOutIndex()));

  auto update  = getInTensor(GatherGradOp::gradInIndex());
  auto indices = getInTensor(GatherGradOp::indicesInIndex());

  auto result = popops::createGatherInput(graph().getPoplarGraph(),
                                          update.elementType(),
                                          outputShape,
                                          static_cast<unsigned>(axis),
                                          popops::GatherParams{},
                                          debugContext("result"));

  // Zero the result tensor
  popops::zero(graph().getPoplarGraph(), result, prog, debugContext("zero"));

  if (result.numElements() == 0 || update.numElements() == 0 ||
      indices.numElements() == 0) {
    setOutTensor(GatherGradOp::gradOutIndex(), result);
    return;
  }

  auto scale = graph().getPoplarGraph().addConstant(
      update.elementType(), {}, 1.0f, debugContext("const_1"));
  graph().getPoplarGraph().setTileMapping(scale, 0);

  // Flatten the index shaped region of the update
  update = update.flatten(static_cast<unsigned>(axis),
                          static_cast<unsigned>(axis) + indices.rank());
  // Put the slice dimension at the front
  update = update.dimRoll(static_cast<unsigned>(axis));
  // Flatten the rest of the dimensions
  update = update.flatten(1, update.rank());
  // Add a degenerate dimension
  update = update.expand({1});

  auto target = result;
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

  // Accumulate the updates into the target
  popops::multiUpdateAdd(graph().getPoplarGraph(),
                         target,
                         update,
                         indices,
                         scale,
                         {0},
                         {1},
                         prog,
                         popops::SlicePlan(),
                         poplar::OptionFlags(),
                         debugContext());

  setOutTensor(GatherGradOp::gradOutIndex(), result);
}

namespace {
OpxCreator<GatherOpx> gatherOpxCreator({Onnx::Operators::Gather_1,
                                        Onnx::Operators::Gather_11});
OpxCreator<GatherGradOpx> gatherGradOpxCreator(Onnx::GradOperators::GatherGrad);
} // namespace

} // namespace popx
} // namespace popart
