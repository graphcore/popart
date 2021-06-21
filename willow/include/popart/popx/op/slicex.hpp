// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SLICEX_HPP
#define GUARD_NEURALNET_SLICEX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {

class SliceOp;
class SliceInplaceOp;

namespace popx {

class BaseSliceOpx : public PopOpx {
public:
  BaseSliceOpx(Op *, Devicex *);

  InputCreatorType getInputCreatorType(InIndex) const final;

  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;

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

#endif
