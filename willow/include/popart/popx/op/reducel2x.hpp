// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCEL2X_HPP
#define GUARD_NEURALNET_REDUCEL2X_HPP

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

class ReduceL2Opx : public PopOpx {
public:
  ReduceL2Opx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

class ReduceL2GradOpx : public PopOpx {
public:
  ReduceL2GradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const override;
};

} // namespace popx
} // namespace popart

#endif
