// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_MINX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_MINX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/names.hpp>

#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class MinOpx : public Opx {
public:
  MinOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const override;

  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
};

class MinArgGradOpx : public Opx {
public:
  MinArgGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_MINX_HPP_
