// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONCATX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONCATX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {

class ConcatOp;
class ConcatGradOp;
class Op;

namespace popx {
class Devicex;

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONCATX_HPP_
