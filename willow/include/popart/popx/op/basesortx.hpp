// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BASESORTX_HPP
#define GUARD_NEURALNET_BASESORTX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

struct FullSortResult {

  FullSortResult(snap::Tensor indices_, snap::Tensor values_, int axis_)
      : indices(indices_), values(values_), axis(axis_) {}

  snap::Tensor indices;
  snap::Tensor values;
  int axis;
};

class BaseSortOpx : public PopOpx {
public:
  BaseSortOpx(Op *, Devicex *);

  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex index0) const final;

protected:
  // sorted values, and indices of sorted values
  FullSortResult growFullSortResult(snap::program::Sequence &prog) const;

  // indices of sorted values
  snap::Tensor growIndicesSort(snap::program::Sequence &prog) const;

  // axis to sort on
  unsigned axis;
};

} // namespace popx
} // namespace popart

#endif
