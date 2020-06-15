// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESIZEX_HPP
#define GUARD_NEURALNET_RESIZEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class ResizeOp;

namespace popx {

class ResizeOpx : public Opx {
public:
  ResizeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor resize_nearest(poplar::Tensor &input, int dim, int size) const;
};

} // namespace popx
} // namespace popart

#endif
