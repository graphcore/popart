// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICZEROX_HPP
#define GUARD_NEURALNET_DYNAMICZEROX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/dynamic/dynamicupdatex.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class DynamicZeroOpx : public PopOpx {
public:
  DynamicZeroOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}
  void grow(snap::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  virtual snap::Tensor cloneNcopyOpt(snap::program::Sequence &,
                                     const snap::Tensor &) const;
};

class DynamicZeroInplaceOpx : public DynamicZeroOpx {
public:
  DynamicZeroInplaceOpx(Op *op, Devicex *devicex)
      : DynamicZeroOpx(op, devicex) {}
  snap::Tensor cloneNcopyOpt(snap::program::Sequence &,
                             const snap::Tensor &) const override;
};

} // namespace popx
} // namespace popart

#endif
