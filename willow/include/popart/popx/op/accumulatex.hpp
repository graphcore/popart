// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ACCUMULATEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ACCUMULATEX_HPP_

#include <set>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <popops/DynamicSlice.hpp>
#include <popart/names.hpp>
#include <popart/popx/op/varupdatex.hpp>

#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;
class ViewChangers;

class AccumulateBaseOpx : public VarUpdateOpx {
public:
  AccumulateBaseOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const override = 0;

  // can create the accumulator input Tensor (@Var index)
  // from the weight gradient tensor (@Updater index)
  poplar::Tensor createInput(InIndex,
                             const poplar::DebugNameAndId &dnai) const override;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const override;

  InputCreatorType getInputCreatorType(InIndex) const final;

  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

class AccumulateOpx final : public AccumulateBaseOpx {
public:
  AccumulateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class RescaleAccumulateOpx final : public AccumulateBaseOpx {
public:
  RescaleAccumulateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SparseAccumulateOpx final : public AccumulateBaseOpx {
public:
  SparseAccumulateOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor createInput(InIndex,
                             const poplar::DebugNameAndId &dnai) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;

private:
  poplar::OptionFlags options;
  popops::SlicePlan plan;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ACCUMULATEX_HPP_
