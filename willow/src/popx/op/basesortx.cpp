// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/op/basesort.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/basesortx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Sort.hpp>
#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

BaseSortOpx::BaseSortOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<BaseSortOp>(op);
  auto baseSort = dynamic_cast<BaseSortOp *>(op);
  axis          = static_cast<unsigned>(baseSort->getAxis());
}

poplar::Tensor
BaseSortOpx::getIotaTensor(poplar::program::Sequence &prog) const {

  const auto &input = getInTensor(BaseSortOp::getInIndex());

  // The number of elements to be sorted per 1-D vector
  const auto sortSize = input.dim(axis);

  // The number of 1-D vectors to be sorted
  const auto nToSort = input.numElements() / sortSize;

  std::vector<int> iotaVals(sortSize);
  std::iota(iotaVals.begin(), iotaVals.end(), 0);

  auto c = graph().addConstant(poplar::INT,
                               {sortSize},
                               poplar::ArrayRef<int>(iotaVals),
                               debugPrefix("sortSize"));
  poputil::mapTensorLinearly(graph(), c);

  // Fill a tensor with [0, 1, 2, ... nToSort-1] along "axis"
  auto indices = graph().clone(poplar::INT, input);
  prog.add(poplar::program::WriteUndef(indices));

  // new view of indices, dim-shuffling the given axis
  // to the back, and making 2-D
  std::vector<unsigned> permutation(indices.rank());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[axis], permutation.back());
  poplar::Tensor shuffledView =
      indices.dimShuffle(permutation).reshape({nToSort, sortSize});

  // Loop over the front dimension and copy in the constant.
  for (int i = 0; i < nToSort; ++i) {
    prog.add(poplar::program::Copy(c, shuffledView[i]));
  }

  return indices;
}

FullSortResult
BaseSortOpx::growFullSortResult(poplar::program::Sequence &prog) const {

  auto input   = getInTensor(BaseSortOp::getInIndex());
  auto values  = cloneNcopy(prog, input);
  auto indices = getIotaTensor(prog);

  // sort indices and values, using values as the "keys" to sort on
  popops::sortKeyValueInPlace(
      graph(), values, indices, axis, prog, debugPrefix());
  return FullSortResult(indices, values, axis);
}

poplar::Tensor
BaseSortOpx::growIndicesSort(poplar::program::Sequence &prog) const {
  auto input   = getInTensor(BaseSortOp::getInIndex());
  auto indices = getIotaTensor(prog);
  return popops::sortKeyValue(
      graph(), input, indices, axis, prog, debugPrefix());
}

poplar::Tensor BaseSortOpx::createInput(InIndex inIndex,
                                        const std::string &name) const {

  if (inIndex == BaseSortOp::getInIndex()) {
    // Create an input that will minimise the amount of exchange in sort. This
    // means minimising the number of tile boundaries on the given axis.

    auto info = inInfo(BaseSortOp::getInIndex());

    // Put the given axis at the back of the shape.
    auto shape = info.shape_szt();
    std::swap(shape[axis], shape.back());

    // Create a new variable of the modified shape
    auto t = graph().addVariable(popType(info), shape, name);

    // Map it linearly
    poputil::mapTensorLinearly(graph(), t);

    // DimShuffle back to the desired shape
    std::vector<unsigned> permutation(t.rank());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[axis], permutation.back());
    return t.dimShuffle(permutation);
  } else {
    return Opx::createInput(inIndex, name);
  }
}

InputCreatorType BaseSortOpx::getInputCreatorType(InIndex inIndex) const {
  if (inIndex == BaseSortOp::getInIndex()) {
    return InputCreatorType::CanCreate;
  } else {
    return Opx::getInputCreatorType(inIndex);
  }
}

std::vector<TensorId> BaseSortOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

} // namespace popx
} // namespace popart
