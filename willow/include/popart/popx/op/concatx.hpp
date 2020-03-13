// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONCATX_HPP
#define GUARD_NEURALNET_CONCATX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class ConcatOp;
class ConcatGradOp;

namespace popx {

class BaseConcatOpx : public Opx {
public:
  BaseConcatOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final;

  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;

  view::RegMap unwindRegion(InIndex, OutIndex) const final;

private:
  const ConcatOp *const op;
};

class ConcatOpx : public BaseConcatOpx {
public:
  ConcatOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  const ConcatOp *const op;
};

class ConcatInplaceOpx : public BaseConcatOpx {
public:
  ConcatInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  const ConcatOp *const op;
};

class ConcatGradOpx : public Opx {
public:
  ConcatGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  const ConcatGradOp *const op;
};

} // namespace popx
} // namespace popart

#endif
