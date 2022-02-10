// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/scatter.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <snap/popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
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
  std::vector<std::size_t> shape(a.rank(), 1);
  const auto b_shape = b.shape();

  std::copy(b_shape.begin(), b_shape.end(), shape.begin() + dim);

  return b.reshape(shape);
}

snap::Tensor broadcastShape(snap::Tensor a, snap::Tensor b_) {
  auto b = b_.getPoplarTensor();
  for (int k = 0; k < a.rank(); ++k) {
    if (b.dim(k) == 1 && a.dim(k) != b.dim(k)) {
      b = b.broadcast(static_cast<unsigned>(a.dim(k)), k);
    }
  }

  return snap::Tensor{b, b_};
}

snap::Tensor linearizeIndices(const PopOpx &opx,
                              snap::program::Sequence &prog,
                              snap::Tensor indices,
                              int numDataCols) {
  // Linearize the indices: map from 2-d indices to 1-d
  auto result     = indices.flatten(1, indices.rank());
  int numCols     = static_cast<int>(result.dim(1));
  auto colIndices = scatterutilx::linspace(opx.graph(),
                                           0,
                                           numCols,
                                           opx.getDebugNameAndId("colIds"),
                                           1,
                                           result.elementType());

  // numDataCols * indices + colIndices
  result                = opx.cloneNcopy(prog, result, "copyIndices");
  auto numDataColsConst = opx.graph().getPoplarGraph().addConstant(
      result.elementType(), {}, numDataCols, opx.getDebugNameAndId("numCols"));
  opx.graph().getPoplarGraph().setTileMapping(numDataColsConst, 0);

  popops::mulInPlace(opx.graph().getPoplarGraph(),
                     result.getPoplarTensor(),
                     numDataColsConst,
                     prog.getPoplarSequence(),
                     opx.getDebugNameAndId("numColsMulIndices"));
  snap::popops::addInPlace(opx.graph(),
                           result,
                           colIndices,
                           prog,
                           opx.getDebugNameAndId("indicesAddColIds"));

  result = result.flatten();
  result = result.expand({1});
  return result;
}

void growScatter(snap::program::Sequence &prog,
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
  std::vector<snap::Tensor> indices_mapped(indices.rank());
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

  popops::scatter(graph.getPoplarGraph(),
                  dataToUpdateInPlace.getPoplarTensor(),
                  vectorizedIndices.getPoplarTensor(),
                  replacementValues.getPoplarTensor(),
                  indices.rank(),
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  prog.getPoplarSequence(),
                  dnai);
}

snap::Tensor growScatterUpdateGrad(const PopOpx &opx,
                                   snap::program::Sequence &prog,
                                   snap::Graph &graph,
                                   const snap::Tensor &gradIn,
                                   const snap::Tensor &indicesIn,
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
    indices      = scatterutilx::linearizeIndices(opx, prog, indices, numCols);
    grad         = grad.flatten();
    grad         = grad.expand({1});
  }

  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);

  auto result = popops::multiSlice(graph.getPoplarGraph(),
                                   grad.getPoplarTensor(),
                                   indices.getPoplarTensor(),
                                   {0},
                                   {1},
                                   prog.getPoplarSequence(),
                                   plan,
                                   poplar::OptionFlags(),
                                   dnai);

  return alignToAxis(snap::Tensor(result, graph), gradOutShape, axis);
}

snap::Tensor growScatterUpdateGrad(snap::program::Sequence &prog,
                                   snap::Graph &graph,
                                   const snap::Tensor &gradIn,
                                   const snap::Tensor &indicesIn,
                                   const popart::TensorInfo &gradOutInfo,
                                   const popops::SlicePlan &plan,
                                   const poplar::DebugNameAndId &dnai) {
  auto indices      = indicesIn.reinterpret(poplar::UNSIGNED_INT);
  auto result       = popops::multiSlice(graph.getPoplarGraph(),
                                   gradIn.getPoplarTensor(),
                                   indices.getPoplarTensor(),
                                   {0},
                                   {1},
                                   prog.getPoplarSequence(),
                                   plan,
                                   poplar::OptionFlags(),
                                   dnai);
  auto result_shape = gradOutInfo.shape_szt();
  return snap::Tensor(result.reshape(result_shape), graph);
}

} // namespace scatterutilx
} // namespace popx
} // namespace popart
