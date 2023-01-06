// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXPANDX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXPANDX_HPP_

#include <cstddef>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class ExpandOp;
class Op;

namespace popx {
class Devicex;

class BaseExpandOpx : public Opx {
protected:
  poplar::Tensor expand_broadcast(const Shape output_shape,
                                  const poplar::Tensor &expand) const;

public:
  BaseExpandOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final;

  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;

  view::RegMap unwindRegion(InIndex, OutIndex) const final;

protected:
  const ExpandOp *const op;
};

class ExpandOpx : public BaseExpandOpx {
public:
  ExpandOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ExpandInplaceOpx : public BaseExpandOpx {
public:
  ExpandInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ExpandGradOpx : public Opx {
public:
  ExpandGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  std::vector<size_t> xShape;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_EXPANDX_HPP_
