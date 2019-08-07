#include <popart/error.hpp>
#include <popart/op/gather.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/gatherx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Scatter.hpp>
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
    // Flatten the scalar indices
    indices = indices.flatten();

    auto result = popops::gather(graph(),
                                 data,
                                 indices.reinterpret(poplar::UNSIGNED_INT),
                                 static_cast<unsigned>(axis),
                                 prog,
                                 popops::GatherParams{},
                                 debugPrefix());

    // Reshape into the expected ONNX shape
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

  auto update = getInTensor(GatherGradOp::gradInIndex());

  auto result = popops::createGatherInput(
      graph(),
      update.elementType(),
      outputShape,
      static_cast<unsigned>(axis),
      popops::GatherParams{},
      op_p->str() + "output" + std::to_string(GatherGradOp::gradOutIndex()));

  auto indices = getInTensor(GatherGradOp::indicesInIndex());

  std::vector<unsigned> update_window_dims(update.rank() - indices.rank());
  auto begin = update_window_dims.begin();
  auto mid   = update_window_dims.begin() + axis;
  auto end   = update_window_dims.end();
  std::iota(begin, mid, 0);
  std::iota(mid, end, axis + indices.rank());

  std::vector<std::size_t> inserted_window_dims = {
      static_cast<std::size_t>(axis)};

  std::vector<unsigned> scatter_dims_to_op = {static_cast<unsigned>(axis)};

  // Add overlapping gradients
  popops::UpdateComputationFunc updateComp =
      [](poplar::Graph &g,
         poplar::Tensor &a,
         poplar::Tensor &b,
         poplar::program::Sequence &p) -> poplar::Tensor {
    popops::addInPlace(g, b, a, p);

    return b;
  };

  // Scatter the grad input into the result
  popops::scatter(graph(),
                  result,
                  indices,
                  update,
                  indices.rank(),
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  updateComp,
                  prog);

  result = result.reshape(outputShape);

  setOutTensor(GatherGradOp::gradOutIndex(), result);
}

namespace {
OpxCreator<GatherOpx> gatherOpxCreator(Onnx::Operators::Gather_1);
OpxCreator<GatherGradOpx> gatherGradOpxCreator(Onnx::GradOperators::GatherGrad);
} // namespace

} // namespace popx
} // namespace popart
