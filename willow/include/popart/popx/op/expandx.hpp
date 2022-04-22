// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONCATX_HPP
#define GUARD_NEURALNET_CONCATX_HPP

#include <cstddef>
#include <snap/Tensor.hpp>
#include <vector>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class ExpandOp;
class Op;

namespace popx {
class Devicex;

class BaseExpandOpx : public PopOpx {
protected:
  snap::Tensor expand_broadcast(const Shape output_shape,
                                const snap::Tensor &expand) const;

public:
  BaseExpandOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final;

  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;

  view::RegMap unwindRegion(InIndex, OutIndex) const final;

protected:
  const ExpandOp *const op;
};

class ExpandOpx : public BaseExpandOpx {
public:
  ExpandOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class ExpandInplaceOpx : public BaseExpandOpx {
public:
  ExpandInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class ExpandGradOpx : public PopOpx {
public:
  ExpandGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  std::vector<size_t> xShape;
};

} // namespace popx
} // namespace popart

#endif
