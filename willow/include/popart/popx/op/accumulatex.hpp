// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATEX_HPP
#define GUARD_NEURALNET_ACCUMULATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class AccumulateBaseOpx : public VarUpdateOpx {
public:
  AccumulateBaseOpx(Op *op, Devicex *devicex) : VarUpdateOpx(op, devicex){};

  // can create the accumulator input Tensor (@Var index)
  // from the weight gradient tensor (@Updater index)
  poplar::Tensor createInput(InIndex,
                             const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;
  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

class AccumulateOpx : public AccumulateBaseOpx {
public:
  AccumulateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class RescaleAccumulateOpx : public AccumulateBaseOpx {
public:
  RescaleAccumulateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
