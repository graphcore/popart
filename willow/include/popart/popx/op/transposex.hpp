// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSPOSEX_HPP
#define GUARD_NEURALNET_TRANSPOSEX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class TransposeOpx : public PopOpx {
public:
  TransposeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class TransposeInplaceOpx : public PopOpx {
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

#endif
