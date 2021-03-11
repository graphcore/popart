// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poputil/TileMapping.hpp>
#include <popart/popx/op/sortutilx.hpp>

namespace popart {
namespace popx {
namespace sortutilx {

poplar::Tensor getIotaTensor(poplar::Graph &graph,
                             const poplar::Tensor &input,
                             unsigned axis,
                             poplar::program::Sequence &prog,
                             const poplar::DebugNameAndId &dnai) {
  // The number of elements to be sorted per 1-D vector
  const auto sortSize = input.dim(axis);

  // The number of 1-D vectors to be sorted
  const auto nToSort = input.numElements() / sortSize;

  std::vector<int> iotaVals(sortSize);
  std::iota(iotaVals.begin(), iotaVals.end(), 0);

  auto singleRowIota = graph.addConstant(poplar::INT,
                                         {sortSize},
                                         poplar::ArrayRef<int>(iotaVals),
                                         {dnai, "constant"});
  poputil::mapTensorLinearly(graph, singleRowIota);

  // Fill a tensor with [0, 1, 2, ... nToSort-1] along "axis"
  auto indices = graph.clone(poplar::INT, input, {dnai, "clone"});
  prog.add(poplar::program::WriteUndef(indices, {dnai, "writeUndef"}));

  // new view of indices, dim-shuffling the given axis
  // to the back, and making 2-D
  std::vector<unsigned> permutation(indices.rank());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[axis], permutation.back());
  poplar::Tensor shuffledView =
      indices.dimShuffle(permutation).reshape({nToSort, sortSize});

  // Loop over the front dimension and copy in the constant.
  for (int i = 0; i < nToSort; ++i) {
    prog.add(poplar::program::Copy(
        singleRowIota, shuffledView[i], false, {dnai, "copy"}));
  }

  return indices;
}

} // namespace sortutilx
} // namespace popx
} // namespace popart
