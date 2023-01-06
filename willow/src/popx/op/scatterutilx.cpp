// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ext/new_allocator.h>
#include <numeric>
#include <vector>
#include <poplar/ArrayRef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Scatter.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/sliceplanx.hpp>

#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
namespace popx {
namespace scatterutilx {

namespace {

poplar::Tensor concat(const std::vector<poplar::Tensor> &ts,
                      unsigned d,
                      poplar::Graph &graph) {
  std::vector<poplar::Tensor> tsP;
  tsP.reserve(ts.size());
  for (auto t : ts) {
    tsP.push_back(t);
  }

  return poplar::concat(tsP, d);
}

} // unnamed namespace

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

poplar::Tensor linearizeIndices(const Opx &opx,
                                poplar::program::Sequence &prog,
                                poplar::Tensor indices,
                                int numDataCols,
                                unsigned group_size) {
  // Linearize the indices: map from 2-d indices to 1-d
  const bool isGrouped        = group_size > 1;
  const unsigned startAxisDim = isGrouped ? 1 : 0;
  auto result     = indices.flatten(startAxisDim + 1, indices.rank());
  int numCols     = static_cast<int>(result.dim(startAxisDim + 1));
  auto colIndices = scatterutilx::linspace(opx.graph(),
                                           0,
                                           numCols,
                                           opx.getDebugNameAndId("colIds"),
                                           1,
                                           result.elementType());

  // numDataCols * indices + colIndices
  result                = opx.cloneNcopy(prog, result, "copyIndices");
  auto numDataColsConst = opx.graph().addConstant(
      result.elementType(), {}, numDataCols, opx.getDebugNameAndId("numCols"));
  opx.graph().setTileMapping(numDataColsConst, 0);

  popops::mulInPlace(opx.graph(),
                     result,
                     numDataColsConst,
                     prog,
                     opx.getDebugNameAndId("numColsMulIndices"));
  popops::addInPlace(opx.graph(),
                     result,
                     colIndices,
                     prog,
                     opx.getDebugNameAndId("indicesAddColIds"));

  std::size_t isGroupedSzt = static_cast<std::size_t>(isGrouped);
  result                   = result.flatten(isGroupedSzt, result.rank());
  result                   = result.expand({1 + isGroupedSzt});
  return result;
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
  for (int i = 0; i < indices_mapped.size(); ++i) {
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

  std::vector<unsigned> update_window_dims(indices_mapped.size());
  std::iota(update_window_dims.begin(), update_window_dims.end(), 0);

  std::vector<std::size_t> inserted_window_dims(indices_mapped.size());
  std::iota(inserted_window_dims.begin(), inserted_window_dims.end(), 0);

  std::vector<unsigned> scatter_dims_to_op(indices_mapped.size());
  std::iota(scatter_dims_to_op.begin(), scatter_dims_to_op.end(), 0);

  auto vectorizedIndices = concat(indices_mapped, indices_mapped.size(), graph);

  popops::scatter(graph,
                  dataToUpdateInPlace,
                  vectorizedIndices,
                  replacementValues,
                  indices.rank(),
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  prog,
                  dnai);
}

poplar::Tensor growScatterUpdateGrad(const Opx &opx,
                                     poplar::program::Sequence &prog,
                                     poplar::Graph &graph,
                                     const poplar::Tensor &gradIn,
                                     const poplar::Tensor &indicesIn,
                                     const popart::Shape &gradOutShape,
                                     int64_t axis,
                                     const popops::SlicePlan &plan,
                                     const poplar::DebugNameAndId &dnai) {
  // Place the gather axis at the front.
  auto grad    = gradIn.dimRoll(axis);
  auto indices = indicesIn.dimRoll(axis);

  if (indices.rank() < 2) {
    indices = indices.expand({1});
    grad    = grad.expand({1});
  } else {
    auto numCols = indices.numElements() / indices.shape().at(0);
    indices = scatterutilx::linearizeIndices(opx, prog, indices, numCols, 1U);
    grad    = grad.flatten();
    grad    = grad.expand({1});
  }

  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);

  auto result = popops::multiSlice(
      graph, grad, indices, {0}, {1}, prog, plan, poplar::OptionFlags(), dnai);

  return alignToAxis(poplar::Tensor(result), gradOutShape, axis, 1U);
}

} // namespace scatterutilx
} // namespace popx
} // namespace popart
