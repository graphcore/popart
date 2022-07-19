// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICUPDATEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICUPDATEX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <set>
#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class DynamicUpdateOpx : public PopOpx {
public:
  DynamicUpdateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex index) const override;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;
  virtual snap::Tensor cloneNcopyOpt(snap::program::Sequence &,
                                     const snap::Tensor &) const;
};

class DynamicUpdateInplaceOpx : public DynamicUpdateOpx {
public:
  DynamicUpdateInplaceOpx(Op *, Devicex *);
  snap::Tensor cloneNcopyOpt(snap::program::Sequence &,
                             const snap::Tensor &) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICUPDATEX_HPP_
