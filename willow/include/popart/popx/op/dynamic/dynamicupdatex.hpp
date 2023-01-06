// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICUPDATEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICUPDATEX_HPP_

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

class DynamicUpdateOpx : public Opx {
public:
  DynamicUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex index) const override;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;
  virtual poplar::Tensor cloneNcopyOpt(poplar::program::Sequence &,
                                       const poplar::Tensor &) const;
};

class DynamicUpdateInplaceOpx : public DynamicUpdateOpx {
public:
  DynamicUpdateInplaceOpx(Op *, Devicex *);
  poplar::Tensor cloneNcopyOpt(poplar::program::Sequence &,
                               const poplar::Tensor &) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICUPDATEX_HPP_
