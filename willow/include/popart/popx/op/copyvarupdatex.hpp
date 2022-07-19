// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_COPYVARUPDATEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_COPYVARUPDATEX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <set>
#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/op/varupdatex.hpp>

#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class CopyVarUpdateOpx : public VarUpdateOpx {
public:
  CopyVarUpdateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // can create updater Tensor from updated Tensor. That is, use the Var Tensor
  // to create the updater.
  snap::Tensor
  createInputTensor(InIndex, const poplar::DebugNameAndId &dnai) const final;

  InputCreatorType getInputCreatorType(InIndex) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_COPYVARUPDATEX_HPP_
