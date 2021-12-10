// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATEX_HPP
#define GUARD_NEURALNET_ACCUMULATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/varupdatex.hpp>

#include <poplar/OptionFlags.hpp>
#include <popops/DynamicSlice.hpp>

namespace popart {
namespace popx {

class AccumulateBaseOpx : public VarUpdateOpx {
public:
  AccumulateBaseOpx(Op *, Devicex *);

  void grow(snap::program::Sequence &) const override = 0;

  // can create the accumulator input Tensor (@Var index)
  // from the weight gradient tensor (@Updater index)
  snap::Tensor
  createInputTensor(InIndex, const poplar::DebugNameAndId &dnai) const override;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const override;

  InputCreatorType getInputCreatorType(InIndex) const final;

  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

class AccumulateOpx final : public AccumulateBaseOpx {
public:
  AccumulateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class RescaleAccumulateOpx final : public AccumulateBaseOpx {
public:
  RescaleAccumulateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class SparseAccumulateOpx final : public AccumulateBaseOpx {
public:
  SparseAccumulateOpx(Op *, Devicex *);

  void grow(snap::program::Sequence &) const final;

  snap::Tensor
  createInputTensor(InIndex, const poplar::DebugNameAndId &dnai) const final;

  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;

private:
  poplar::OptionFlags options;
  popops::SlicePlan plan;
};

} // namespace popx
} // namespace popart

#endif
