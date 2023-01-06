// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_TANHX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_TANHX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_TANHX_HPP_
