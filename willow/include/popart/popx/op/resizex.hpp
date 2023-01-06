// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_RESIZEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_RESIZEX_HPP_

#include <cstdint>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

struct ResizeParams {
  Shape inShape;
  Shape outShape;
  std::vector<float> scales;
  ResizeMode mode;
  ResizeNearestMode nearestMode;
  ResizeCoordinateTransformationMode coordinateTransformationMode;
};

class ResizeOpx : public Opx {
public:
  ResizeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ResizeGradOpx : public Opx {
public:
  ResizeGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  poplar::Tensor resizeNearestGrad(ResizeGradOp &op,
                                   const poplar::Tensor &input,
                                   ResizeParams &params,
                                   poplar::program::Sequence &prog) const;
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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_RESIZEX_HPP_
