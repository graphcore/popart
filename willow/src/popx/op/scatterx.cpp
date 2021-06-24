// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/scatter.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/scatterx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/Cast.hpp>
#include <popops/Gather.hpp>
#include <popops/Scatter.hpp>
#include <poputil/TileMapping.hpp>

#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

ScatterOpx::ScatterOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ScatterOp>(
      op, {Onnx::Operators::Scatter_9, Onnx::Operators::Scatter_11});

  axis = dynamic_cast<ScatterOp *>(op)->getAxis();
}

void ScatterOpx::grow(poplar::program::Sequence &prog) const {
  auto indices = getInTensor(ScatterOp::indicesInIndex()).getPoplarTensor();
  auto data    = cloneNcopy(prog, getInTensor(ScatterOp::dataInIndex()));
  auto values  = getInTensor(ScatterOp::updatesInIndex()).getPoplarTensor();
  scatterutilx::growScatter(prog,
                            graph(),
                            indices,
                            values,
                            data.getPoplarTensor(),
                            axis,
                            getDebugNameAndId("scatter"));
  setOutTensor(ScatterOp::outIndex(), data);
}

ScatterDataGradOpx::ScatterDataGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ScatterDataGradOp>(op, Onnx::GradOperators::ScatterDataGrad);

  axis = dynamic_cast<ScatterDataGradOp *>(op)->getAxis();
}

void ScatterDataGradOpx::grow(poplar::program::Sequence &prog) const {
  auto data = cloneNcopy(prog, getInTensor(ScatterDataGradOp::gradInIndex()))
                  .getPoplarTensor();
  auto indices =
      getInTensor(ScatterDataGradOp::indicesInIndex()).getPoplarTensor();
  auto update = graph().getPoplarGraph().addConstant(
      data.elementType(), indices.shape(), 0, debugContext("zeros"));
  poputil::mapTensorLinearly(graph().getPoplarGraph(), update);

  // Build the implicit index coordinates
  //
  // popops::scatter requires the indices to be complete coordinates into the
  // data tensor, but ONNX scatter only provides an axis and a scalar index.
  std::vector<poplar::Tensor> indices_mapped(indices.rank());
  for (int i = 0; i < indices.rank(); ++i) {
    auto t = scatterutilx::linspace(graph(),
                                    0,
                                    static_cast<int>(indices.dim(i)),
                                    getDebugNameAndId("linspace"));

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
  popops::scatter(graph().getPoplarGraph(),
                  data,
                  indices,
                  update,
                  indices.rank() - 1,
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  prog,
                  debugContext("scatter"));

  setOutTensor(ScatterDataGradOp::gradOutIndex(), snap::Tensor{data, graph()});
}

ScatterUpdateGradOpx::ScatterUpdateGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ScatterUpdateGradOp>(op, Onnx::GradOperators::ScatterUpdateGrad);

  axis = dynamic_cast<ScatterUpdateGradOp *>(op)->getAxis();
}

void ScatterUpdateGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradIn =
      getInTensor(ScatterUpdateGradOp::gradInIndex()).getPoplarTensor();
  auto indices =
      getInTensor(ScatterDataGradOp::indicesInIndex()).getPoplarTensor();

  auto gradOut = scatterutilx::growScatterUpdateGrad(
      prog,
      graph(),
      gradIn,
      indices,
      axis,
      getDebugNameAndId("scatter_update_grad"));

  setOutTensor(ScatterUpdateGradOp::gradOutIndex(),
               snap::Tensor{gradOut, graph()});
}

namespace {
OpxCreator<ScatterOpx> scatterOpxCreator({Onnx::Operators::Scatter_9,
                                          Onnx::Operators::Scatter_11});
OpxCreator<ScatterDataGradOpx>
    scatterDataGradOpxCreator(Onnx::GradOperators::ScatterDataGrad);
OpxCreator<ScatterUpdateGradOpx>
    scatterUpdateGradOpxCreator(Onnx::GradOperators::ScatterUpdateGrad);
} // namespace

} // namespace popx
} // namespace popart
