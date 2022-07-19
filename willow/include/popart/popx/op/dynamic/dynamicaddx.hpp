// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICADDX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICADDX_HPP_

#include <snap/Tensor.hpp>
#include <popart/popx/op/dynamic/dynamicupdatex.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class DynamicAddOpx : public DynamicUpdateOpx {
public:
  DynamicAddOpx(Op *op, Devicex *devicex) : DynamicUpdateOpx(op, devicex) {}
  void grow(snap::program::Sequence &) const final;
};

class DynamicAddInplaceOpx : public DynamicAddOpx {
public:
  DynamicAddInplaceOpx(Op *op, Devicex *devicex) : DynamicAddOpx(op, devicex) {}
  snap::Tensor cloneNcopyOpt(snap::program::Sequence &,
                             const snap::Tensor &) const override;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_DYNAMIC_DYNAMICADDX_HPP_
