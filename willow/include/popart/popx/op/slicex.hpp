// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SLICEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SLICEX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/popx/opx.hpp>

#include "popart/names.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {

class SliceOp;
class SliceInplaceOp;
class Op;

namespace popx {
class Devicex;

class BaseSliceOpx : public Opx {
public:
  BaseSliceOpx(Op *, Devicex *);

  InputCreatorType getInputCreatorType(InIndex) const final;

  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;

  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class SliceOpx : public BaseSliceOpx {
public:
  SliceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  SliceOp *getSliceOp() const;
};

class SliceInplaceOpx : public BaseSliceOpx {
public:
  SliceInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  SliceInplaceOp *getSliceInplaceOp() const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SLICEX_HPP_
