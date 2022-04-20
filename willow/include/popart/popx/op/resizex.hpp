// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RESIZEX_HPP
#define GUARD_NEURALNET_RESIZEX_HPP

#include <cstdint>
#include <snap/Tensor.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class ResizeOpx : public PopOpx {
public:
  ResizeOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class ResizeGradOpx : public PopOpx {
public:
  ResizeGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  snap::Tensor reduceDimension(snap::program::Sequence &,
                               const snap::Tensor &input,
                               int dimension,
                               float scale) const;
  snap::Tensor padDimension(snap::program::Sequence &,
                            const snap::Tensor &input,
                            int dimension,
                            int64_t newSize,
                            float scale) const;
};

} // namespace popx
} // namespace popart

#endif
