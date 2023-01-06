// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICADDX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICADDX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/popx/op/dynamic/dynamicupdatex.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICADDX_HPP_
