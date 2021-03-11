// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/op/basesort.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/basesortx.hpp>
#include <popart/popx/op/sortutilx.hpp>
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

FullSortResult
BaseSortOpx::growFullSortResult(poplar::program::Sequence &prog) const {

  auto input   = getInTensor(BaseSortOp::getInIndex());
  auto values  = cloneNcopy(prog, input);
  auto indices = sortutilx::getIotaTensor(
      graph(), input, axis, prog, getDebugNameAndId("iotaTensor"));

  // sort indices and values, using values as the "keys" to sort on
  popops::sortKeyValueInPlace(
      graph(), values, indices, axis, prog, debugContext("sort"));
  return FullSortResult(indices, values, axis);
}

poplar::Tensor
BaseSortOpx::growIndicesSort(poplar::program::Sequence &prog) const {
  auto input   = getInTensor(BaseSortOp::getInIndex());
  auto indices = sortutilx::getIotaTensor(
      graph(), input, axis, prog, getDebugNameAndId("iotaTensor"));
  return popops::sortKeyValue(
      graph(), input, indices, axis, prog, debugContext("sort"));
}

poplar::Tensor
BaseSortOpx::createInput(InIndex inIndex,
                         const poplar::DebugNameAndId &dnai) const {

  if (inIndex == BaseSortOp::getInIndex()) {
    // Create an input that will minimise the amount of exchange in sort. This
    // means minimising the number of tile boundaries on the given axis.

    auto info = inInfo(BaseSortOp::getInIndex());

    // Put the given axis at the back of the shape.
    auto shape = info.shape_szt();
    std::swap(shape[axis], shape.back());

    // Create a new variable of the modified shape
    auto t = graph().addVariable(popType(info), shape, dnai);

    // Map it linearly
    poputil::mapTensorLinearly(graph(), t);

    // DimShuffle back to the desired shape
    std::vector<unsigned> permutation(t.rank());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[axis], permutation.back());
    return t.dimShuffle(permutation);
  } else {
    return Opx::createInput(inIndex, dnai);
  }
}

InputCreatorType BaseSortOpx::getInputCreatorType(InIndex inIndex) const {
  if (inIndex == BaseSortOp::getInIndex()) {
    return InputCreatorType::CanCreate;
  } else {
    return Opx::getInputCreatorType(inIndex);
  }
}

std::set<TensorId> BaseSortOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

} // namespace popx
} // namespace popart
