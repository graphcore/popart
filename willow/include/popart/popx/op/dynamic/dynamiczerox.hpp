// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICZEROX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICZEROX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class DynamicZeroOpx : public Opx {
public:
  DynamicZeroOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}
  void grow(poplar::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  virtual poplar::Tensor cloneNcopyOpt(poplar::program::Sequence &,
                                       const poplar::Tensor &) const;
};

class DynamicZeroInplaceOpx : public DynamicZeroOpx {
public:
  DynamicZeroInplaceOpx(Op *op, Devicex *devicex)
      : DynamicZeroOpx(op, devicex) {}
  poplar::Tensor cloneNcopyOpt(poplar::program::Sequence &,
                               const poplar::Tensor &) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICZEROX_HPP_
