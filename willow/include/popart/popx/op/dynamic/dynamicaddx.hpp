// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICADDX_HPP
#define GUARD_NEURALNET_DYNAMICADDX_HPP

#include <popart/names.hpp>
#include <popart/popx/op/dynamic/dynamicupdatex.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class DynamicAddOpx : public DynamicUpdateOpx {
public:
  DynamicAddOpx(Op *op, Devicex *devicex) : DynamicUpdateOpx(op, devicex) {}
  void grow(poplar::program::Sequence &) const final;
};

class DynamicAddInplaceOpx : public DynamicAddOpx {
public:
  DynamicAddInplaceOpx(Op *op, Devicex *devicex) : DynamicAddOpx(op, devicex) {}
  poplar::Tensor cloneNcopyOpt(poplar::program::Sequence &,
                               const poplar::Tensor &) const override;
};

} // namespace popx
} // namespace popart

#endif
