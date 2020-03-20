// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COPYVARUPDATEX_HPP
#define GUARD_NEURALNET_COPYVARUPDATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class CopyVarUpdateOpx : public VarUpdateOpx {
public:
  CopyVarUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // can create updater Tensor from updated Tensor. That is, use the Var Tensor
  // to create the updater.
  poplar::Tensor createInput(InIndex,
                             const std::string &updaterName) const final;

  InputCreatorType getInputCreatorType(InIndex) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
