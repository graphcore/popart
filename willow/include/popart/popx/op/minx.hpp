// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MINX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MINX_HPP_

#include <snap/Tensor.hpp>
#include <popart/names.hpp>

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

class MinOpx : public PopOpx {
public:
  MinOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const override;

  snap::Tensor
      unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

class MinArgGradOpx : public PopOpx {
public:
  MinArgGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_MINX_HPP_
