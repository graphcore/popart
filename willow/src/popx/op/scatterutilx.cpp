// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/scatter.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/Cast.hpp>
#include <popops/Gather.hpp>
#include <popops/Scatter.hpp>
#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {
namespace scatterutilx {

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

snap::Tensor linspace(snap::Graph &graph,
                      int left,
                      int right,
                      const poplar::DebugNameAndId &dnai,
                      int increment,
                      const poplar::Type &type) {
  std::size_t count = right - left;

  std::vector<int> values(count);
  std::iota(values.begin(), values.end(), 0);
  std::transform(values.begin(),
                 values.end(),
                 values.begin(),
                 [left, increment](int v) { return left + v * increment; });

  auto result = graph.getPoplarGraph().addConstant(
      type, {count}, poplar::ArrayRef<int>(values), {dnai, "count"});

  graph.getPoplarGraph().setTileMapping(result, 0);

  return snap::Tensor{result, graph};
}

snap::Tensor matchRank(snap::Tensor a, snap::Tensor b, unsigned dim) {
  std::vector<std::size_t> shape(a.getPoplarTensor().rank(), 1);
  const auto b_shape = b.getPoplarTensor().shape();

  std::copy(b_shape.begin(), b_shape.end(), shape.begin() + dim);

  return snap::Tensor{b.getPoplarTensor().reshape(shape), b};
}

snap::Tensor broadcastShape(snap::Tensor a, snap::Tensor b_) {
  auto b = b_.getPoplarTensor();
  for (int k = 0; k < a.getPoplarTensor().rank(); ++k) {
    if (b.dim(k) == 1 && a.getPoplarTensor().dim(k) != b.dim(k)) {
      b = b.broadcast(static_cast<unsigned>(a.getPoplarTensor().dim(k)), k);
    }
  }

  return snap::Tensor{b, b_};
}

void growScatter(poplar::program::Sequence &prog,
                 snap::Graph &graph,
                 const snap::Tensor &indices,
                 const snap::Tensor &replacementValues,
                 const snap::Tensor &dataToUpdateInPlace,
                 int64_t axis,
                 const poplar::DebugNameAndId &dnai) {
  // Build the implicit index coordinates
  //
  // popops::scatter requires the indices to be complete coordinates into the
  // data tensor, but ONNX scatter only provides an axis and a scalar index.
  std::vector<snap::Tensor> indices_mapped(indices.getPoplarTensor().rank());
  for (int i = 0; i < indices_mapped.size(); ++i) {
    auto t = linspace(graph,
                      0,
                      static_cast<int>(indices.getPoplarTensor().dim(i)),
                      {dnai, "linspace"},
                      1,
                      indices.getPoplarTensor().elementType());

    // Match the rank of indices
    t = matchRank(indices, t, i);

    // Match the shape of indices
    indices_mapped[i] = broadcastShape(indices, t);
  }

  // Replace the axis indices with the user provided indices
  indices_mapped[axis] = indices;

  // Add a degenerate dimension for concatenation
  for (auto &index : indices_mapped) {
    index = snap::Tensor{
        index.getPoplarTensor().expand({index.getPoplarTensor().rank()}),
        graph};
  }

  std::vector<unsigned> update_window_dims(indices_mapped.size());
  std::iota(update_window_dims.begin(), update_window_dims.end(), 0);

  std::vector<std::size_t> inserted_window_dims(indices_mapped.size());
  std::iota(inserted_window_dims.begin(), inserted_window_dims.end(), 0);

  std::vector<unsigned> scatter_dims_to_op(indices_mapped.size());
  std::iota(scatter_dims_to_op.begin(), scatter_dims_to_op.end(), 0);

  auto vectorizedIndices = concat(indices_mapped, indices_mapped.size(), graph);

  popops::scatter(graph.getPoplarGraph(),
                  dataToUpdateInPlace.getPoplarTensor(),
                  vectorizedIndices.getPoplarTensor(),
                  replacementValues.getPoplarTensor(),
                  indices.getPoplarTensor().rank(),
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  prog,
                  dnai);
}

snap::Tensor growScatterUpdateGrad(poplar::program::Sequence &prog,
                                   snap::Graph &graph,
                                   const snap::Tensor &gradIn,
                                   const snap::Tensor &indices,
                                   int64_t axis,
                                   const poplar::DebugNameAndId &dnai) {
  // Build the implicit index coordinates
  //
  // Create a grid of linspaced indices
  // Start by creating 1D linspaced constant tensors
  std::vector<snap::Tensor> indicesMapped(gradIn.getPoplarTensor().rank());
  for (int i = 0; i < indicesMapped.size(); ++i) {
    indicesMapped[i] =
        linspace(graph,
                 0,
                 static_cast<int>(indices.getPoplarTensor().dim(i)),
                 {dnai, "linspace"},
                 1,
                 indices.getPoplarTensor().elementType());
  }

  // Match the rank of the indices to the update tensor
  for (int i = 0; i < indicesMapped.size(); ++i) {
    indicesMapped[i] = matchRank(indices, indicesMapped[i], i);
  }

  for (auto &index : indicesMapped) {
    // Match the shape of update
    index = broadcastShape(indices, index);
  }

  // Replace the axis indices with the user provided indices
  indicesMapped[axis] = indices;

  for (auto &index : indicesMapped) {
    // Add a degenerate dimension for concatenation
    index = snap::Tensor{
        index.getPoplarTensor().expand({index.getPoplarTensor().rank()}),
        graph};
  }

  // Concat the indices on the degenerate dimension
  auto indicesGrid =
      concat(indicesMapped, indices.getPoplarTensor().rank(), graph);
  indicesGrid = indicesGrid.reinterpret(poplar::UNSIGNED_INT);

  const auto indexVectorDim = indicesGrid.getPoplarTensor().rank() - 1;
  std::vector<std::size_t> sliceSizes(indicesMapped.size(), 1);

  std::vector<std::size_t> collapsedSliceDims(indicesMapped.size());
  std::iota(collapsedSliceDims.begin(), collapsedSliceDims.end(), 0);

  std::vector<unsigned> startIndexMap(indicesGrid.getPoplarTensor().rank() - 1);
  std::iota(startIndexMap.begin(), startIndexMap.end(), 0);

  // Gather the elements from the grad input
  return snap::Tensor{popops::gather(graph.getPoplarGraph(),
                                     gradIn.getPoplarTensor(),
                                     indicesGrid.getPoplarTensor(),
                                     indexVectorDim,
                                     {},
                                     sliceSizes,
                                     collapsedSliceDims,
                                     startIndexMap,
                                     prog,
                                     {dnai, "gather"}),
                      graph};
}

} // namespace scatterutilx
} // namespace popx
} // namespace popart
