// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1ACCUMULATETHEACCLX_HPP
#define GUARD_NEURALNET_SGD1ACCUMULATETHEACCLX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class SGD1AccumulateOpx : public VarUpdateOpx {
public:
  SGD1AccumulateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // can create the accumulator input Tensor (@Var index)
  // from the weight gradient tensor (@Updater index)
  poplar::Tensor createInput(InIndex, const std::string &name) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
