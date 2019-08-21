#include <popart/error.hpp>
#include <popart/op/gather.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/gatherx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>

namespace popart {
namespace popx {

GatherOpx::GatherOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GatherOp>(op, Onnx::Operators::Gather_1);

  axis = dynamic_cast<GatherOp *>(op)->getAxis();
}

void GatherOpx::grow(poplar::program::Sequence &prog) const {
  const auto indicesShape = inShape(GatherOp::indicesInIndex());
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(GatherOp::outIndex()));

  auto indices = getInTensor(GatherOp::indicesInIndex());
  auto data    = getInTensor(GatherOp::dataInIndex());

  // If there are no indices, return an empty tensor of the appropriate
  // shape
  if (indices.numElements() == 0) {
    auto result = graph().addVariable(
        data.elementType(), outputShape, debugPrefix("result"));

    setOutTensor(GatherOp::outIndex(), result);
  } else {
    // Flatten the scalar indices.
    auto offsets = indices.flatten();
    // Add a degenerate dimension at the end.
    offsets = offsets.expand({1});
    // reinterpret the indices as unsigned int. This assumes negative indices.
    // are impossible.
    offsets = offsets.reinterpret(poplar::UNSIGNED_INT);

    // Create a permutation that swaps the gather axis for the front.
    std::vector<unsigned> permutation(data.rank(), 0);
    boost::iota(permutation, 0);
    std::swap(permutation.front(), permutation[axis]);

    // Place the gather axis at the front.
    data = data.dimShuffle(permutation);
    // Store the shape for later.
    auto tmp_shape = data.shape();
    // Flatten the other dimensions.
    data = data.flatten(1, data.rank());

    auto result = popops::multiSlice(
        graph(), data, offsets, {0}, {1}, prog, debugPrefix());

    // Reshape the result to "unflatten" the other dimensions.
    tmp_shape.front() = result.dim(0);
    result            = result.reshape(tmp_shape);
    // Put the gather axis dimension back in the right place.
    result = result.dimShuffle(permutation);

    // Reshape into the expected ONNX shape.
    result = result.reshape(outputShape);

    setOutTensor(GatherOp::outIndex(), result);
  }
}

poplar::Tensor GatherOpx::createInput(int index,
                                      const std::string &name) const {
  if (index != GatherOp::dataInIndex()) {
    throw error("GatherOpx::createInput Cannot create input {}", index);
  }

  auto info        = inInfo(GatherOp::dataInIndex());
  const auto shape = info.shape_szt();

  return popops::createGatherInput(graph(),
                                   popType(info),
                                   shape,
                                   static_cast<unsigned>(axis),
                                   popops::GatherParams{},
                                   name);
}

InputCreatorType GatherOpx::getInputCreatorType(int index0) const {
  return index0 == GatherOp::dataInIndex() ? InputCreatorType::CANCREATE
                                           : Opx::getInputCreatorType(index0);
}

bool GatherOpx::createsEquiv(int, const Opx *, int) const { return false; }

std::vector<TensorId> GatherOpx::mustExistBeforeCreate(int) const { return {}; }

GatherGradOpx::GatherGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<GatherGradOp>(op, Onnx::GradOperators::GatherGrad);

  axis = dynamic_cast<GatherGradOp *>(op)->getAxis();
}

void GatherGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto outputShape =
      vXtoY<int64_t, std::size_t>(outShape(GatherGradOp::gradOutIndex()));

  auto update  = getInTensor(GatherGradOp::gradInIndex());
  auto indices = getInTensor(GatherGradOp::indicesInIndex());

  auto result = popops::createGatherInput(graph(),
                                          update.elementType(),
                                          outputShape,
                                          static_cast<unsigned>(axis),
                                          popops::GatherParams{},
                                          debugPrefix("result"));

  // Zero the result tensor
  popops::zero(graph(), result, prog, debugPrefix("zero"));

  if (result.numElements() == 0 || update.numElements() == 0 ||
      indices.numElements() == 0) {
    setOutTensor(GatherGradOp::gradOutIndex(), result);
    return;
  }

  auto scale = graph().addConstant(
      update.elementType(), {}, 1.0f, debugPrefix("const_1"));
  graph().setTileMapping(scale, 0);

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
  popops::multiUpdateAdd(
      graph(), target, update, indices, scale, {0}, {1}, prog, debugPrefix());

  setOutTensor(GatherGradOp::gradOutIndex(), result);
}

namespace {
OpxCreator<GatherOpx> gatherOpxCreator(Onnx::Operators::Gather_1);
OpxCreator<GatherGradOpx> gatherGradOpxCreator(Onnx::GradOperators::GatherGrad);
} // namespace

} // namespace popx
} // namespace popart
