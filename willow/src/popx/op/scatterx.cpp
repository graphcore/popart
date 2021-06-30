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

namespace {

snap::Tensor
concat(const std::vector<snap::Tensor> &ts, unsigned d, snap::Graph &graph) {
  std::vector<poplar::Tensor> tsP;
  tsP.reserve(ts.size());
  for (auto t : ts) {
    tsP.push_back(t.getPoplarTensor());
  }

  return snap::Tensor{poplar::concat(tsP, d), graph};
}

} // unnamed namespace

ScatterOpx::ScatterOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ScatterOp>(
      op, {Onnx::Operators::Scatter_9, Onnx::Operators::Scatter_11});

  axis = dynamic_cast<ScatterOp *>(op)->getAxis();
}

void ScatterOpx::grow(poplar::program::Sequence &prog) const {
  auto indices = getInTensor(ScatterOp::indicesInIndex());
  auto data    = cloneNcopy(prog, getInTensor(ScatterOp::dataInIndex()));
  auto values  = getInTensor(ScatterOp::updatesInIndex());
  scatterutilx::growScatter(
      prog, graph(), indices, values, data, axis, getDebugNameAndId("scatter"));
  setOutTensor(ScatterOp::outIndex(), data);
}

ScatterDataGradOpx::ScatterDataGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ScatterDataGradOp>(op, Onnx::GradOperators::ScatterDataGrad);

  axis = dynamic_cast<ScatterDataGradOp *>(op)->getAxis();
}

void ScatterDataGradOpx::grow(poplar::program::Sequence &prog) const {
  auto data = cloneNcopy(prog, getInTensor(ScatterDataGradOp::gradInIndex()));
  auto indices = getInTensor(ScatterDataGradOp::indicesInIndex());
  auto update  = snap::Tensor{
      graph().getPoplarGraph().addConstant(data.getPoplarTensor().elementType(),
                                           indices.getPoplarTensor().shape(),
                                           0,
                                           debugContext("zeros")),
      graph()};
  poputil::mapTensorLinearly(graph().getPoplarGraph(),
                             update.getPoplarTensor());

  // Build the implicit index coordinates
  //
  // popops::scatter requires the indices to be complete coordinates into the
  // data tensor, but ONNX scatter only provides an axis and a scalar index.
  std::vector<snap::Tensor> indices_mapped(indices.getPoplarTensor().rank());
  for (int i = 0; i < indices_mapped.size(); ++i) {
    auto t = scatterutilx::linspace(
        graph(),
        0,
        static_cast<int>(indices.getPoplarTensor().dim(i)),
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
    index = snap::Tensor{
        index.getPoplarTensor().expand({index.getPoplarTensor().rank()}),
        graph()};
  }

  std::vector<unsigned> update_window_dims(indices_mapped.size());
  std::iota(update_window_dims.begin(), update_window_dims.end(), 0);

  std::vector<std::size_t> inserted_window_dims(indices_mapped.size());
  std::iota(inserted_window_dims.begin(), inserted_window_dims.end(), 0);

  std::vector<unsigned> scatter_dims_to_op(indices_mapped.size());
  std::iota(scatter_dims_to_op.begin(), scatter_dims_to_op.end(), 0);

  // Concat the indices on the degenerate dimension
  indices = concat(indices_mapped, indices_mapped.size(), graph());

  // Scatter the zeros into data
  popops::scatter(graph().getPoplarGraph(),
                  data.getPoplarTensor(),
                  indices.getPoplarTensor(),
                  update.getPoplarTensor(),
                  indices.getPoplarTensor().rank() - 1,
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  prog,
                  debugContext("scatter"));

  setOutTensor(ScatterDataGradOp::gradOutIndex(), data);
}

ScatterUpdateGradOpx::ScatterUpdateGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ScatterUpdateGradOp>(op, Onnx::GradOperators::ScatterUpdateGrad);

  axis = dynamic_cast<ScatterUpdateGradOp *>(op)->getAxis();
}

void ScatterUpdateGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradIn  = getInTensor(ScatterUpdateGradOp::gradInIndex());
  auto indices = getInTensor(ScatterDataGradOp::indicesInIndex());

  auto gradOut = scatterutilx::growScatterUpdateGrad(
      prog,
      graph(),
      gradIn,
      indices,
      axis,
      getDebugNameAndId("scatter_update_grad"));

  setOutTensor(ScatterUpdateGradOp::gradOutIndex(), gradOut);
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
