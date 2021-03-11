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

poplar::Tensor linspace(poplar::Graph &graph,
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

  auto result = graph.addConstant(
      type, {count}, poplar::ArrayRef<int>(values), {dnai, "count"});

  graph.setTileMapping(result, 0);

  return result;
}

poplar::Tensor matchRank(poplar::Tensor a, poplar::Tensor b, unsigned dim) {
  std::vector<std::size_t> shape(a.rank(), 1);
  const auto b_shape = b.shape();

  std::copy(b_shape.begin(), b_shape.end(), shape.begin() + dim);

  return b.reshape(shape);
}

poplar::Tensor broadcastShape(poplar::Tensor a, poplar::Tensor b) {
  for (int k = 0; k < a.rank(); ++k) {
    if (b.dim(k) == 1 && a.dim(k) != b.dim(k)) {
      b = b.broadcast(static_cast<unsigned>(a.dim(k)), k);
    }
  }

  return b;
}

void growScatter(poplar::program::Sequence &prog,
                 poplar::Graph &graph,
                 const poplar::Tensor &indices,
                 const poplar::Tensor &replacementValues,
                 const poplar::Tensor &dataToUpdateInPlace,
                 int64_t axis,
                 const poplar::DebugNameAndId &dnai) {
  // Build the implicit index coordinates
  //
  // popops::scatter requires the indices to be complete coordinates into the
  // data tensor, but ONNX scatter only provides an axis and a scalar index.
  std::vector<poplar::Tensor> indices_mapped(indices.rank());
  for (int i = 0; i < indices.rank(); ++i) {
    auto t = linspace(graph,
                      0,
                      static_cast<int>(indices.dim(i)),
                      {dnai, "linspace"},
                      1,
                      indices.elementType());

    // Match the rank of indices
    t = matchRank(indices, t, i);

    // Match the shape of indices
    indices_mapped[i] = broadcastShape(indices, t);
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

  auto vectorizedIndices = poplar::concat(indices_mapped, indices.rank());

  popops::scatter(graph,
                  dataToUpdateInPlace,
                  vectorizedIndices,
                  replacementValues,
                  indices.rank(),
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  prog);
}

} // namespace scatterutilx
} // namespace popx
} // namespace popart
