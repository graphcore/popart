// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONCATX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONCATX_HPP_

#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {

class ConcatOp;
class ConcatGradOp;
class Op;

namespace popx {
class Devicex;

class BaseConcatOpx : public PopOpx {
public:
  BaseConcatOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final;

  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;

  view::RegMap unwindRegion(InIndex, OutIndex) const final;

private:
  const ConcatOp *const op;
};

class ConcatOpx : public BaseConcatOpx {
public:
  ConcatOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  const ConcatOp *const op;
};

class ConcatInplaceOpx : public BaseConcatOpx {
public:
  ConcatInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  const ConcatOp *const op;
};

class ConcatGradOpx : public PopOpx {
public:
  ConcatGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  const ConcatGradOp *const op;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CONCATX_HPP_
