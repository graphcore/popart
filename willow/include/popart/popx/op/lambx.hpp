// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LAMBX_HPP
#define GUARD_NEURALNET_LAMBX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class LambSquareOpx : public Opx {
public:
  LambSquareOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class LambR2SquareOpx : public Opx {
public:
  LambR2SquareOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
