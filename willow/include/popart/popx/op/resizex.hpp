// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESIZEX_HPP
#define GUARD_NEURALNET_RESIZEX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class ResizeOpx : public PopOpx {
public:
  ResizeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor
  resizeDim(poplar::Tensor &input, int dim, int64_t size, float scale) const;

  poplar::Tensor resizeNearestNeighbour(poplar::Tensor &input,
                                        int dim,
                                        int64_t size,
                                        float scale) const;

  float coordinateTransformation(float idx, int dim) const;
  int64_t applyNearestMode(float idx) const;
};

class ResizeGradOpx : public PopOpx {
public:
  ResizeGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor reduceDimension(poplar::program::Sequence &,
                                 const poplar::Tensor &input,
                                 int dimension,
                                 float scale) const;
  poplar::Tensor padDimension(poplar::program::Sequence &,
                              const poplar::Tensor &input,
                              int dimension,
                              int64_t newSize,
                              float scale) const;
};

} // namespace popx
} // namespace popart

#endif
