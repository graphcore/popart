// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_TRANSPOSEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_TRANSPOSEX_HPP_

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

class TransposeOpx : public Opx {
public:
  TransposeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class TransposeInplaceOpx : public Opx {
public:
  TransposeInplaceOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final;
  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class TransposeGradOpx : public TransposeOpx {
public:
  TransposeGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_TRANSPOSEX_HPP_
