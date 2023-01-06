// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_BASESORTX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_BASESORTX_HPP_

#include <set>
#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

struct FullSortResult {

  FullSortResult(poplar::Tensor indices_, poplar::Tensor values_, int axis_)
      : indices(indices_), values(values_), axis(axis_) {}

  poplar::Tensor indices;
  poplar::Tensor values;
  int axis;
};

class BaseSortOpx : public Opx {
public:
  BaseSortOpx(Op *, Devicex *);

  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex index0) const final;

protected:
  // sorted values, and indices of sorted values
  FullSortResult growFullSortResult(poplar::program::Sequence &prog) const;

  // indices of sorted values
  poplar::Tensor growIndicesSort(poplar::program::Sequence &prog) const;

  // axis to sort on
  unsigned axis;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_BASESORTX_HPP_
