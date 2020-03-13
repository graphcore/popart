// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TANHX_HPP
#define GUARD_NEURALNET_TANHX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class TanhOpx : public Opx {
public:
  TanhOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class TanhGradOpx : public Opx {
public:
  TanhGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
