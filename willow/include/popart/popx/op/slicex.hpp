// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SLICEX_HPP
#define GUARD_NEURALNET_SLICEX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

class SliceOp;
class SliceInplaceOp;

namespace popx {

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

class SliceGradOpx : public Opx {
public:
  SliceGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  std::pair<bool, poplar::Tensor> getPreSlicedTensorIfPossible() const;
};

} // namespace popx
} // namespace popart

#endif
