#include <poponnx/error.hpp>
#include <poponnx/op/scatter.hpp>
#include <poponnx/popx/op/scatterutilx.hpp>
#include <poponnx/popx/op/scatterx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/util.hpp>

#include <popops/Cast.hpp>
#include <popops/Gather.hpp>
#include <popops/Scatter.hpp>
#include <poputil/TileMapping.hpp>

#include <poputil/TileMapping.hpp>

namespace poponnx {
namespace popx {

ScatterOpx::ScatterOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ScatterOp>(op, Onnx::Operators::Scatter_9);

  axis = dynamic_cast<ScatterOp *>(op)->getAxis();
}

void ScatterOpx::grow(poplar::program::Sequence &prog) const {
  auto indices = getInTensor(ScatterOp::indicesInIndex());
  auto data    = cloneNcopy(prog, getInTensor(ScatterOp::dataInIndex()));
  auto values  = getInTensor(ScatterOp::updatesInIndex());
  scatterutilx::growScatter(prog, graph(), indices, values, data, axis);
  setOutTensor(ScatterOp::outIndex(), data);
}

ScatterDataGradOpx::ScatterDataGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ScatterDataGradOp>(op, Onnx::GradOperators::ScatterDataGrad);

  axis = dynamic_cast<ScatterDataGradOp *>(op)->getAxis();
}

void ScatterDataGradOpx::grow(poplar::program::Sequence &prog) const {
  auto data = cloneNcopy(prog, getInTensor(ScatterDataGradOp::gradInIndex()));
  auto indices = getInTensor(ScatterDataGradOp::indicesInIndex());
  auto update  = graph().addConstant(
      data.elementType(), indices.shape(), 0, debugPrefix("zeros"));
  poputil::mapTensorLinearly(graph(), update);

  // Build the implicit index coordinates
  //
  // popops::scatter requires the indices to be complete coordinates into the
  // data tensor, but ONNX scatter only provides an axis and a scalar index.
  std::vector<poplar::Tensor> indices_mapped(indices.rank());
  for (int i = 0; i < indices.rank(); ++i) {
    auto t =
        scatterutilx::linspace(graph(), 0, static_cast<int>(indices.dim(i)));

    // Match the rank of indices
    t = scatterutilx::matchRank(indices, t, i);

    // Match the shape of indices
    indices_mapped[i] = scatterutilx::broadcastShape(indices, t);
  }

  // Replace the axis indices with the user provided indices
  indices_mapped[axis] = indices;

  // Add a degenerate dimension for concatenation
  for (auto &index : indices_mapped) {
    index = index.expand({index.rank()});
  }

  std::vector<unsigned> update_window_dims(indices.rank());
  std::iota(update_window_dims.begin(), update_window_dims.end(), 0);

  std::vector<std::size_t> inserted_window_dims(indices.rank());
  std::iota(inserted_window_dims.begin(), inserted_window_dims.end(), 0);

  std::vector<unsigned> scatter_dims_to_op(indices.rank());
  std::iota(scatter_dims_to_op.begin(), scatter_dims_to_op.end(), 0);

  // Concat the indices on the degenerate dimension
  indices = poplar::concat(indices_mapped, indices.rank());

  // Scatter the zeros into data
  popops::scatter(graph(),
                  data,
                  indices,
                  update,
                  indices.rank() - 1,
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  prog);

  setOutTensor(ScatterDataGradOp::gradOutIndex(), data);
}

ScatterUpdateGradOpx::ScatterUpdateGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ScatterUpdateGradOp>(op, Onnx::GradOperators::ScatterUpdateGrad);

  axis = dynamic_cast<ScatterUpdateGradOp *>(op)->getAxis();
}

void ScatterUpdateGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto gradIn = getInTensor(ScatterUpdateGradOp::gradInIndex());
  auto indices      = getInTensor(ScatterDataGradOp::indicesInIndex());

  // Build the implicit index coordinates
  //
  // Create a grid of linspaced indices
  // Start by creating 1D linspaced constant tensors
  std::vector<poplar::Tensor> indices_mapped(gradIn.rank());
  for (int i = 0; i < gradIn.rank(); ++i) {
    indices_mapped[i] =
        scatterutilx::linspace(graph(), 0, static_cast<int>(indices.dim(i)));
  }

  // Match the rank of the indices to the update tensor
  for (int i = 0; i < gradIn.rank(); ++i) {
    indices_mapped[i] = scatterutilx::matchRank(indices, indices_mapped[i], i);
  }

  for (auto &index : indices_mapped) {
    // Match the shape of update
    index = scatterutilx::broadcastShape(indices, index);
  }

  // Replace the axis indices with the user provided indices
  indices_mapped[axis] = indices;

  for (auto &index : indices_mapped) {
    // Add a degenerate dimension for concatenation
    index = index.expand({index.rank()});
  }

  // Concat the indices on the degenerate dimension
  indices = poplar::concat(indices_mapped, indices.rank());

  const auto index_vector_dim = indices.rank() - 1;
  std::vector<std::size_t> sliceSizes(gradIn.rank(), 1);

  std::vector<std::size_t> collapsedSliceDims(gradIn.rank());
  std::iota(collapsedSliceDims.begin(), collapsedSliceDims.end(), 0);

  std::vector<unsigned> startIndexMap(indices.rank() - 1);
  std::iota(startIndexMap.begin(), startIndexMap.end(), 0);

  // Gather the elements from the grad input
  auto result = popops::gather(graph(),
                               gradIn,
                               indices,
                               index_vector_dim,
                               {},
                               sliceSizes,
                               collapsedSliceDims,
                               startIndexMap,
                               prog);

  setOutTensor(ScatterDataGradOp::gradOutIndex(), result);
}

namespace {
OpxCreator<ScatterOpx> scatterOpxCreator(Onnx::Operators::Scatter_9);
OpxCreator<ScatterDataGradOpx>
    scatterDataGradOpxCreator(Onnx::GradOperators::ScatterDataGrad);
OpxCreator<ScatterUpdateGradOpx>
    scatterUpdateGradOpxCreator(Onnx::GradOperators::ScatterUpdateGrad);
} // namespace

} // namespace popx
} // namespace poponnx
