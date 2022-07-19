// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MAXX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MAXX_HPP_

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

// Refactor needed, see T7199
class MaxOpx : public PopOpx {
public:
  MaxOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const override;

  snap::Tensor
      unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

class MaxArgGradOpx : public PopOpx {
public:
  MaxArgGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_MAXX_HPP_
