// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICZEROX_HPP
#define GUARD_NEURALNET_DYNAMICZEROX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/dynamic/dynamicupdatex.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class DynamicZeroOp;

namespace popx {

class DynamicZeroOpx : public DynamicUpdateOpx {
public:
  DynamicZeroOpx(Op *op, Devicex *devicex) : DynamicUpdateOpx(op, devicex) {}
  void grow(poplar::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex index) const final;
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

#endif
