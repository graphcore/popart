// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCESUMSQUAREX_HPP
#define GUARD_NEURALNET_REDUCESUMSQUAREX_HPP

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

class ReduceSumSquareOpx : public PopOpx {
public:
  ReduceSumSquareOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceSumSquareGradOpx : public PopOpx {
public:
  ReduceSumSquareGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
